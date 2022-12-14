from pprint import pprint
from utils import get_datasets
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from models.argmax.atom import AtomFlow
from models.pos.flow import TransformerCoorFlow, TransformerCoorFlowV2
from models.pos.distro_base import BaseNet
from larsflow.distributions import ResampledGaussian
from models.classifier import PosClassifier

import torch
import numpy as np
from einops import rearrange
from xyz2mol.xyz2mol import xyz2mol

atom_decoder = ['H', 'C', 'N', 'O', 'F']

@torch.jit.script
def remove_mean_with_constraint(x, size_constraint):
    mean = torch.sum(x, dim=1, keepdim=True) / size_constraint
    x = x - mean
    return x

if __name__ == "__main__":
    base_normal = torch.distributions.Normal(loc=0., scale=1.)

    resampled = True
    batch_size = 1000

    coor_net = TransformerCoorFlowV2(
        hidden_dim=128,
        num_layers_transformer=8,
        block_size=6,
        max_nodes=18,
        conv1x1=True,
        conv1x1_node_wise=True,
        batch_norm=False,
        act_norm=True,
        partition_size=(1,6),
        squeeze=True,
        squeeze_step=2
    )

    states = torch.load("outputs/model_checkpoint_1mghyk3o_2550.pt", map_location="cpu")
    
    coor_net.load_state_dict(
        states['model_state_dict']
    )

    if resampled:
        net = BaseNet(
            hidden_dim=128,
            num_layers=8,
            max_nodes=18,
            n_dim=3,
        )

        base = ResampledGaussian(
            d=18 * 3,
            a=net,
            T=100,
            eps=0.1,
            trainable=True
        )

        base.load_state_dict(
            states['base']
        )
        base.eval()
        

    print("Loaded TransformerCoorFlow model...")

    print("Sampling...")

    mol_size = 18

    if resampled:
        z, _ = base.forward(num_samples=batch_size)
        z = rearrange(z, "b (d n) -> b d n", d=3)
    else:
        z = torch.randn(batch_size, mol_size, 3,)
        z = remove_mean_with_constraint(z, mol_size)
        z = rearrange(z, "b d n -> b n d")


    print(z.shape)
    with torch.no_grad():
        pos, _ = coor_net.inverse(
            z,
        )

    print("Sampled Positions...")
    print(pos.shape)

    classifier = PosClassifier(feats_dim=64, hidden_dim=256, gnn_size=5)
    classifier.load_state_dict(torch.load("classifier.pt", map_location="cpu")['model_state_dict'])



    net = AtomFlow(
        hidden_dim=32,
        block_size=6,
        encoder_size=4,
        gnn_size=2,
        num_classes=5,
        stochastic_permute=False
    )

    net.load_state_dict(
        torch.load("outputs/model_checkpoint_3pchowk4_200.pt", map_location="cpu")['model_state_dict']
    )

    print("Loaded AtomFlow model...")


    zeros = torch.zeros(batch_size, 3, 29 - mol_size)
    pos = torch.cat([pos, zeros], dim=2)
    pos = rearrange(pos, "b d n -> b n d")
    mask = torch.ones(batch_size, 29, dtype=torch.bool)
    mask[:, mol_size:] = False

    with torch.no_grad():
        output = torch.sigmoid(classifier(pos, mask=mask)).squeeze()
        print(output.sum())

    
    with torch.no_grad():
        atoms_types, _ =net.inverse(
            base_normal.sample(sample_shape=(pos.shape[0], 29, 5)) * mask.unsqueeze(2),
            pos,
            mask = mask
        )

    print("Sampled Atom Types...")
    valid = 0

    atoms_types_n = atoms_types.long().numpy()
    pos_n = pos.numpy()

    valid_smiles =[]
    valid_mols = []
    valid_idx = []

    invalid_idx = []

    for idx in range(atoms_types.shape[0]):
        size = mask[idx].to(torch.long).sum()
        atom_decoder_int = [1, 6, 7, 8, 9]
        atom_ty =[atom_decoder_int[i] for i in atoms_types_n[idx, :size]]

        pos_t = pos_n[idx, :size].tolist()


        try:
            mols = xyz2mol(
                atom_ty,
                pos_t,
                use_huckel=True,
            )


            for mol in mols:
                smiles = Chem.MolToSmiles(mol)

               
                valid += 1
                valid_idx.append(idx)
                valid_smiles.append(smiles)
                valid_mols.append(mol)
                break
        except:
            invalid_idx.append(idx)


    pprint(valid_smiles)
    print(valid * 1.0 / batch_size)
    

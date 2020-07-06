#!/usr/bin/env python3
"""
Script to build graphs
"""
from histocartography.graph_generation.arg_parser_graph_build import arg_parser
from histocartography.utils.io import get_device, read_params
from histocartography.graph_generation.cell_graphs.build_cell_graph import BuildCellGraph
from histocartography.graph_generation.superpx_graphs.build_superpx_graph import BuildSPGraph
from histocartography.graph_generation.assignment_matrices.build_assignment_matrix import BuildAssignmnentMatrix
import torch
import os


def main(args):

    config = read_params(args.configuration, verbose=True)

    if config['model_type'] == 'cell_graph_model':
        graph_cell = BuildCellGraph(args, 'cpu')
        graph_cell.run()

    if config['model_type'] == 'superpx_graph_model':
        graph_superpx = BuildSPGraph(args, 'cpu')
        graph_superpx.run()

    if config['model_type'] == 'multi_level_graph_model':
    	graph_assignment = BuildAssignmnentMatrix(args, 'cpu')
    	graph_assignment.run()


if __name__ == "__main__":
    main(args=arg_parser())

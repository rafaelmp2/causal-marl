import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from acd_model.modules import *
from acd_model.MLPEncoder import MLPEncoder
from acd_model.CNNEncoder import CNNEncoder
from acd_model.MLPEncoderUnobserved import MLPEncoderUnobserved
from acd_model.EncoderGlobalTemp import CNNEncoderGlobalTemp

from acd_model.MLPDecoder import MLPDecoder
from acd_model.RNNDecoder import RNNDecoder
from acd_model.SimulationDecoder import SimulationDecoder
from acd_model.DecoderGlobalTemp import MLPDecoderGlobalTemp, SimulationDecoderGlobalTemp

from acd_model import utils


def load_distribution(args):
    edge_probs = torch.randn(
        torch.Size([args.num_atoms ** 2 - args.num_atoms, args.edge_types]),
        device=args.device.type,
        requires_grad=True,
    )
    return edge_probs


def load_encoder(args):
    if args.global_temp:
        encoder = CNNEncoderGlobalTemp(
            args,
            args.dims,
            args.encoder_hidden,
            args.edge_types,
            args.encoder_dropout,
            args.factor,
        )
    elif args.unobserved > 0 and args.model_unobserved == 0:
        encoder = MLPEncoderUnobserved(
            args,
            args.timesteps * args.dims,
            args.encoder_hidden,
            args.edge_types,
            do_prob=args.encoder_dropout,
            factor=args.factor,
        )
    else:
        # here
        if args.encoder == "mlp":
            encoder = MLPEncoder(
                args,
                args.timesteps * args.dims,
                args.encoder_hidden,
                args.edge_types,
                do_prob=args.encoder_dropout,
                factor=args.factor,
            )
        elif args.encoder == "cnn":
            encoder = CNNEncoder(
                args,
                args.dims,
                args.encoder_hidden,
                args.edge_types,
                args.encoder_dropout,
                args.factor,
            )

    encoder, num_GPU = utils.distribute_over_GPUs(args, encoder, num_GPU=args.num_GPU)
    if args.load_folder:
        print("Loading model file")
        args.encoder_file = os.path.join(args.load_folder, "encoder_150.pt")
        encoder.load_state_dict(torch.load(args.encoder_file, map_location=args.device))

    return encoder


def load_decoder(args, loc_max, loc_min, vel_max, vel_min):
    if args.global_temp:
        if args.decoder == "mlp":
            decoder = MLPDecoderGlobalTemp(
                n_in_node=args.dims,
                edge_types=args.edge_types,
                msg_hid=args.decoder_hidden,
                msg_out=args.decoder_hidden,
                n_hid=args.decoder_hidden,
                do_prob=args.decoder_dropout,
                skip_first=args.skip_first,
                latent_dim=args.latent_dim,
            )
        elif args.decoder == "sim":
            decoder = SimulationDecoderGlobalTemp(
                loc_max, loc_min, vel_max, vel_min, args.suffix
            )
    else:
        if args.decoder == "mlp":
            decoder = MLPDecoder(
                args,
                n_in_node=args.dims,
                edge_types=args.edge_types,
                msg_hid=args.decoder_hidden,
                msg_out=args.decoder_hidden,
                n_hid=args.decoder_hidden,
                do_prob=args.decoder_dropout,
                skip_first=args.skip_first,
            )
        elif args.decoder == "rnn":
            decoder = RNNDecoder(
                n_in_node=args.dims,
                edge_types=args.edge_types,
                n_hid=args.decoder_hidden,
                do_prob=args.decoder_dropout,
                skip_first=args.skip_first,
            )
        elif args.decoder == "sim":
            decoder = SimulationDecoder(loc_max, loc_min, vel_max, vel_min, args.suffix)

    decoder, num_GPU = utils.distribute_over_GPUs(args, decoder, num_GPU=args.num_GPU)
    # print("Let's use", num_GPU, "GPUs!")

    if args.load_folder:
        print("Loading model file")
        args.decoder_file = os.path.join(args.load_folder, "decoder.pt")
        decoder.load_state_dict(torch.load(args.decoder_file, map_location=args.device))
        args.save_folder = False

    return decoder


def load_model(args, loc_max, loc_min, vel_max, vel_min):

    if args.use_encoder:
        encoder = load_encoder(args)
    else:
        encoder = None
        edge_probs = load_distribution(args)
        optimizer = optim.Adam(
            [{"params": edge_probs, "lr": args.lr_z}]
            + [{"params": decoder.parameters(), "lr": args.lr}]
        )

    return (
        encoder,
        None,
        None,
        None,
        None,
    )

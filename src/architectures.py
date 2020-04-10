from fairseq.models import register_model_architecture
from fairseq.models.fconv import base_architecture


# @register_model_architecture('fconv', 'fconv_jp_current')
# def fconv_jp_current(args):
#     convs = '[(192, 3)] * 3'
#     convs += ' + [(384, 3)] * 3'
#     convs += ' + [(768, 3)] * 2'
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 192)
#     args.encoder_layers = getattr(args, 'encoder_layers', convs)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 192)
#     args.decoder_layers = getattr(args, 'decoder_layers', convs)
#     args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 192)
#     args.dropout = getattr(args, 'dropout', 0.15)
#     args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
#     args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
#     args.decoder_attention = getattr(args, 'decoder_attention', 'True')
#     args.share_input_output_embed = getattr(
#         args, 'share_input_output_embed', False)
#     base_architecture(args)


# @register_model_architecture('fconv', 'fconv_jp_mini')
# def fconv_jp_mini(args):
#     convs = '[(192, 3)] * 2'
#     convs += ' + [(384, 3)] * 2'
#     convs += ' + [(768, 3)] * 1'
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
#     args.encoder_layers = getattr(args, 'encoder_layers', convs)
#     args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 128)
#     args.decoder_layers = getattr(args, 'decoder_layers', convs)
#     args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 192)
#     args.dropout = getattr(args, 'dropout', 0.15)
#     args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
#     args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
#     args.decoder_attention = getattr(args, 'decoder_attention', 'True')
#     args.share_input_output_embed = getattr(
#         args, 'share_input_output_embed', False)
#     base_architecture(args)

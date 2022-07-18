import torch
import torch.nn as nn
import torchvision.models as models
import pprint
import torch.nn.functional as F

class SGR_Wrapper(nn.Module):
    def __init__(self, sgr_model, t_head=5):
        super(SGR_Wrapper, self).__init__()
        self.nf_enc = sgr_model.nf_enc
        self.start_aux = sgr_model.start_aux
        self.norm_input = sgr_model.norm_input
        self.encoder = sgr_model.encoder
        self.feature_learning = nn.ModuleList()
        self.heads = sgr_model.heads[:t_head]
        self.feature_learning = sgr_model.feature_learning[:t_head-1]
        n_decoder_heads = sgr_model.start_aux + t_head

        if n_decoder_heads < len(sgr_model.decoder):
            self.decoder = sgr_model.decoder[:n_decoder_heads]
            self.post_layers = []
        else:
            self.decoder = sgr_model.decoder
            n_left_heads = n_decoder_heads - len(sgr_model.decoder)
            self.post_layers = sgr_model.post_layers[:n_left_heads]

    def forward(self, x, early_exit=None):
        size = (x.size(2), x.size(3))
        # Normalize the input
        x = self.norm_input(x)
        input_org = x  # will be used by last post_layer via skip-connect

        # Add layers in the encoder (E1-4)
        enc_outputs = self.encoder(x)
        enc_outputs = enc_outputs[1:]
        x = enc_outputs[-1]
        # Add layers in the decoder (D1-4)
        n_encoder_layers = len(self.nf_enc)
        cnt = 0

        for (idx, layer) in enumerate(self.decoder):
            # concatenate the output with the corresponding encoder layer if it is required
            if idx != 0:  # concatenate on the channel dimension
                enc_output = enc_outputs[n_encoder_layers - idx - 1]
                #enc_output = self.skip_conns[-idx](x_dec=x, x_enc=enc_output)
                x = torch.cat((x, enc_output), 1)

            # Add the decoder layer
            x = layer(x)
            # Student sub-networks
            if idx == self.start_aux:
                o = self.heads[cnt](x)
            elif idx > self.start_aux:
                feat_size = (x.size(2), x.size(3))
                x, a = self.feature_learning[cnt](x,F.interpolate(o,feat_size))
                cnt += 1
                o = self.heads[cnt](x)
            # if early_exit == cnt and idx >= self.start_aux:
            #     return o

        # Add layers after the decoder (D5 and D6)
        for (idx, layer) in enumerate(self.post_layers):
            # Concatenate the input of second last layer with the original
            enc_output = None
            if idx == 0:
                enc_output = input_org

            # Add the layer
            x = layer(x)
            if idx + n_encoder_layers == self.start_aux:
                o = self.heads[cnt](x)
            elif idx + n_encoder_layers > self.start_aux:
                feat_size = (x.size(2), x.size(3))
                x, a = self.feature_learning[cnt](x, F.interpolate(o, feat_size))
                cnt += 1
                o = self.heads[cnt](x)

        # # Recovery the size of the input
        # x = self.recovery_size(x, self.norm_input.padding)
        return x, F.interpolate(o, size=size)




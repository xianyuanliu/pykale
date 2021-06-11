from torch import nn
from torchvision.models.utils import load_state_dict_from_url

from kale.embed.video_selayer import SELayer4feat
from kale.embed.video_transformer import TransformerBlock

model_urls = {
    "rgb_boring": None,
    "flow_boring": None,
    "audio_boring": None,
}


class BoringNetVideo(nn.Module):
    """Regular simple network for video input.

    Args:
        input_size (int, optional): the dimension of the final feature vector. Defaults to 512.
        n_channel (int, optional): the number of channel for Linear and BN layers.
        dropout_keep_prob (int, optional): the dropout probability for keeping the parameters.
    """

    def __init__(self, input_size=512, n_channel=512, n_out=256, dropout_keep_prob=0.5):
        super(BoringNetVideo, self).__init__()
        self.hidden_sizes = 512
        self.num_layers = 4

        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim=input_size,
                    num_heads=8,
                    att_dropout=0.1,
                    att_resid_dropout=0.1,
                    final_dropout=0.1,
                    max_seq_len=9,
                    ff_dim=self.hidden_sizes,
                    causal=False,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.fc1 = nn.Linear(input_size, n_channel)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(dropout_keep_prob)
        self.fc2 = nn.Linear(n_channel, n_out)
        self.selayer1 = SELayer4feat(channel=8, reduction=2)

        # self.dim_reduction_layer = torch.nn.Identity()
        #
        # self.classification_vector = nn.Parameter(torch.randn(1, 1, input_size))
        # self.pos_encoding = nn.Parameter(
        #     torch.randn(1, 9, input_size)
        # )

    def forward(self, x):
        # (B, F, INPUT_DIM) -> (B, F, D)

        # x = self.dim_reduction_layer(x)
        # B, F, D = x.size()

        # classification_vector = self.classification_vector.repeat((B, 1, 1))
        # (B, F, D) -> (B, 1+F, D)
        # x = torch.cat([classification_vector, x], dim=1)
        # seq_len = x.size(1)
        # for layer in self.transformer:
        #     x = x + self.pos_encoding[:, :seq_len, :]
        #     x = layer(x)
        x = self.fc2(self.dp1(self.relu1(self.fc1(x))))
        x = self.selayer1(x)
        return x


def boring_net(name, pretrained=False, input_size=1024, n_out=256, progress=True):
    model = BoringNetVideo(input_size=input_size, n_out=n_out)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[name], progress=progress)
        model.load_state_dict(state_dict)
    return model


def boring_net_joint(
    rgb_name=None, flow_name=None, audio_name=None, pretrained=False, input_size=1024, n_out=256, progress=True
):
    model_rgb = model_flow = model_audio = None
    if rgb_name is not None:
        model_rgb = boring_net(rgb_name, pretrained, input_size, n_out, progress)
    if flow_name is not None:
        model_flow = boring_net(flow_name, pretrained, input_size, n_out, progress)
    if audio_name is not None:
        model_audio = boring_net(audio_name, pretrained, input_size, n_out, progress)
    return {"rgb": model_rgb, "flow": model_flow, "audio": model_audio}

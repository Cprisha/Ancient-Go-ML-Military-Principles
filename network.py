import numpy as np


def im2col_indices(x, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x.shape

    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    x_pad = np.pad(
        x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
    )

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)

    i1 = stride * np.repeat(np.arange(out_height), out_width)

    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(
        np.arange(C), field_height * field_width
    ).reshape(-1, 1)

    cols = x_pad[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(
        field_height * field_width * C, -1
    )

    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):

    N, C, H, W = x_shape

    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    x_pad = np.zeros(
        (N, C, H_padded, W_padded), dtype=cols.dtype
    )

    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)

    i1 = stride * np.repeat(np.arange(out_height), out_width)

    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(
        np.arange(C), field_height * field_width
    ).reshape(-1, 1)

    cols_reshaped = cols.reshape(
        C * field_height * field_width, -1, N
    )

    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    np.add.at(
        x_pad, (slice(None), k, i, j), cols_reshaped
    )

    if padding == 0:
        return x_pad

    return x_pad[:, :, padding:-padding, padding:-padding]


class GoConv2D:
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1, stride=1):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        scale = np.sqrt(
            2.0 / (in_channels * kernel_size * kernel_size)
        )

        self.W = (
            np.random.randn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size
            ) * scale
        )

        self.b = np.zeros((out_channels, 1))

        self.params = [self.W, self.b]
        self.grads = [
            np.zeros_like(self.W),
            np.zeros_like(self.b)
        ]

        self.cache = None


    def forward(self, x):
        N, C, H, W = x.shape

        cols = im2col_indices(
            x, self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride
        )

        W_row = self.W.reshape(self.out_channels, -1)

        out = np.dot(W_row, cols) + self.b

        out_h = int(
            (H + 2 * self.padding - self.kernel_size) / self.stride + 1
        )

        out_w = int(
            (W + 2 * self.padding - self.kernel_size) / self.stride + 1
        )

        out = out.reshape(
            self.out_channels, out_h, out_w, N
        ).transpose(3, 0, 1, 2)

        self.cache = (x, cols)

        return out


    def backward(self, dout):
        x, cols = self.cache

        dout_reshaped = (
            dout.transpose(1, 2, 3, 0)
            .reshape(self.out_channels, -1)
        )

        self.grads[0] = np.dot(
            dout_reshaped,
            cols.T
        ).reshape(self.W.shape)

        self.grads[1] = np.sum(
            dout_reshaped,
            axis=1,
            keepdims=True
        )

        W_row = self.W.reshape(self.out_channels, -1)

        dcols = np.dot(
            W_row.T,
            dout_reshaped
        )

        dx = col2im_indices(
            dcols,
            x.shape,
            self.kernel_size,
            self.kernel_size,
            padding=self.padding,
            stride=self.stride
        )

        return dx


class GoReLU:
    def __init__(self):
        self.params = []
        self.grads = []
        self.cache = None


    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)


    def backward(self, dout):
        return dout * (self.cache > 0)


class GoResBlock:
    def __init__(self, channels):

        self.conv1 = GoConv2D(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.relu1 = GoReLU()

        self.conv2 = GoConv2D(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.relu2 = GoReLU()

        self.params = (
            self.conv1.params +
            self.conv2.params
        )

        self.grads = (
            self.conv1.grads +
            self.conv2.grads
        )


    def forward(self, x):

        out = self.conv1.forward(x)
        out = self.relu1.forward(out)

        out = self.conv2.forward(out)


        out = self.relu2.forward(out + x)

        return out


    def backward(self, dout):

        dout_after_relu = self.relu2.backward(dout)

        dx = self.conv2.backward(dout_after_relu)
        dx = self.relu1.backward(dx)
        dx = self.conv1.backward(dx)

        return dx + dout_after_relu


class GoDense:
    def __init__(self, in_features, out_features):

        scale = np.sqrt(2.0 / in_features)

        self.W = (
            np.random.randn(in_features, out_features)
            * scale
        )

        self.b = np.zeros((1, out_features))

        self.params = [self.W, self.b]

        self.grads = [
            np.zeros_like(self.W),
            np.zeros_like(self.b)
        ]

        self.cache = None


    def forward(self, x):

        self.cache = x

        return np.dot(x, self.W) + self.b


    def backward(self, dout):

        x = self.cache

        self.grads[0] = np.dot(x.T, dout)

        self.grads[1] = np.sum(
            dout,
            axis=0,
            keepdims=True
        )

        return np.dot(dout, self.W.T)


class GoEmbedding:
    def __init__(self, num_embeddings, embedding_dim):

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.W = (
            np.random.randn(
                num_embeddings,
                embedding_dim
            ) * 0.01
        )

        self.params = [self.W]

        self.grads = [
            np.zeros_like(self.W)
        ]

        self.cache = None


    def forward(self, ids):

        self.cache = ids

        return self.W[ids]


    def backward(self, dout):

        ids = self.cache

        self.grads[0].fill(0.0)


        np.add.at(
            self.grads[0],
            ids,
            dout
        )

        return None


class GoGlobalAvgPool2D:
    def __init__(self):

        self.params = []
        self.grads = []

        self.cache = None


    def forward(self, x):

        self.cache = x.shape

        return np.mean(x, axis=(2, 3))


    def backward(self, dout):

        N, C, H, W = self.cache

        dout_expanded = dout[:, :, None, None]

        return (
            np.broadcast_to(
                dout_expanded,
                (N, C, H, W)
            )
            / (H * W)
        )


class PolicyNet:

    def __init__(self, in_channels=6, num_res_blocks=6):

        self.stem = GoConv2D(
            in_channels,
            64,
            kernel_size=3,
            padding=1
        )

        self.stem_relu = GoReLU()

        self.res_blocks = [
            GoResBlock(64)
            for _ in range(num_res_blocks)
        ]

        self.policy_conv = GoConv2D(
            64,
            1,
            kernel_size=1,
            padding=0
        )

        self.pass_dense = GoDense(
            64 * 19 * 19,
            1
        )

        self.layers = (
            [self.stem, self.stem_relu]
            + self.res_blocks
            + [self.policy_conv, self.pass_dense]
        )


    def forward(self, x):

        out = self.stem.forward(x)
        out = self.stem_relu.forward(out)

        for block in self.res_blocks:
            out = block.forward(out)

        spatial_logits = (
            self.policy_conv.forward(out)
            .reshape(x.shape[0], 361)
        )

        pass_logit = self.pass_dense.forward(
            out.reshape(x.shape[0], -1)
        )

        return np.hstack([
            spatial_logits,
            pass_logit
        ])


    def get_params_and_grads(self):

        params = []
        grads = []

        for layer in self.layers:

            params.extend(layer.params)
            grads.extend(layer.grads)

        return params, grads


class CandidateNet:
    def __init__(
        self,
        in_channels=7,
        num_res_blocks=6,
        num_principles=281
    ):

        self.stem = GoConv2D(
            in_channels,
            64,
            kernel_size=3,
            padding=1
        )

        self.stem_relu = GoReLU()

        self.res_blocks = [
            GoResBlock(64)
            for _ in range(num_res_blocks)
        ]

        self.gap = GoGlobalAvgPool2D()


        self.num_dense1 = GoDense(31, 64)
        self.num_relu1 = GoReLU()

        self.num_dense2 = GoDense(64, 64)
        self.num_relu2 = GoReLU()



        self.embedding = GoEmbedding(
            num_principles,
            32
        )


        
        self.alignment_head = GoDense(
            160,
            num_principles
        )

        self.confidence_head = GoDense(
            160,
            num_principles
        )

        self.value_head = GoDense(
            160,
            1
        )


        self.layers = (
            [self.stem, self.stem_relu]
            + self.res_blocks
            + [
                self.gap,

                self.num_dense1,
                self.num_relu1,

                self.num_dense2,
                self.num_relu2,

                self.embedding,

                self.alignment_head,
                self.confidence_head,
                self.value_head
            ]
        )


    def forward(
        self,
        candidate_spatial,
        numeric_features,
        principle_ids
    ):

        out = self.stem.forward(candidate_spatial)
        out = self.stem_relu.forward(out)

        for block in self.res_blocks:
            out = block.forward(out)

        spatial_vec = self.gap.forward(out)


        num_vec = self.num_dense1.forward(
            numeric_features
        )

        num_vec = self.num_relu1.forward(
            num_vec
        )

        num_vec = self.num_dense2.forward(
            num_vec
        )

        num_vec = self.num_relu2.forward(
            num_vec
        )


        emb_vec = self.embedding.forward(
            principle_ids
        )


        fused = np.hstack([
            spatial_vec,
            num_vec,
            emb_vec
        ])


        alignment = self.alignment_head.forward(
            fused
        )

        confidence = self.confidence_head.forward(
            fused
        )

        value = self.value_head.forward(
            fused
        )

        return alignment, confidence, value


    def get_params_and_grads(self):

        params = []
        grads = []

        for layer in self.layers:

            params.extend(layer.params)
            grads.extend(layer.grads)

        return params, grads


def board_to_spatial_planes(
    board_state,
    current_player,
    candidate_move=None
):
    

    num_channels = (
        7
        if candidate_move is not None
        else 6
    )

    planes = np.zeros(
        (num_channels, 19, 19),
        dtype=np.float32
    )

    friendly = (
        1
        if current_player == "B"
        else -1
    )

    opponent = (
        -1
        if current_player == "B"
        else 1
    )


    planes[0] = (
        board_state == friendly
    ).astype(np.float32)

    planes[1] = (
        board_state == opponent
    ).astype(np.float32)



    planes[2] = (
        1.0
        if current_player == "B"
        else 0.0
    )


    if (
        candidate_move is not None
        and candidate_move != "PASS"
    ):

        r, c = candidate_move

        planes[6, r, c] = 1.0


    return planes


def masked_policy_loss(
    logits,
    targets,
    legal_mask
):


    N = logits.shape[0]

    masked_logits = np.where(
        legal_mask,
        logits,
        -1e9
    )



    shifted = (
        masked_logits
        - np.max(
            masked_logits,
            axis=-1,
            keepdims=True
        )
    )

    probs = np.exp(shifted)

    probs /= np.sum(
        probs,
        axis=-1,
        keepdims=True
    )


    loss = -np.sum(
        np.log(
            probs[np.arange(N), targets]
            + 1e-10
        )
    ) / N


    dlogits = probs.copy()

    dlogits[
        np.arange(N),
        targets
    ] -= 1.0


    dlogits = (
        dlogits
        * legal_mask
        / N
    )

    return loss, dlogits


def sparse_principle_loss(
    preds,
    targets,
    principle_mask
):


    diff = (
        preds - targets
    ) * principle_mask


    num_labels = (
        np.sum(principle_mask)
        + 1e-8
    )


    loss = (
        np.sum(diff ** 2)
        / num_labels
    )

    dpreds = (
        2.0 * diff
        / num_labels
    )


    return loss, dpreds


class NumPyAdam:
    def __init__(
        self,
        params,
        grads,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        max_norm=5.0
    ):

        self.params = params
        self.grads = grads

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.eps = eps
        self.max_norm = max_norm

        self.t = 0


        self.m = [
            np.zeros_like(p)
            for p in self.params
        ]

        self.v = [
            np.zeros_like(p)
            for p in self.params
        ]


    def step(self):

        self.t += 1


        total_norm = np.sqrt(
            sum(
                np.sum(g ** 2)
                for g in self.grads
            )
        )


        clip_coef = (
            self.max_norm
            / (total_norm + 1e-6)
        )


        if clip_coef < 1.0:

            for g in self.grads:
                g *= clip_coef


        for i in range(len(self.params)):

            self.m[i] = (
                self.beta1 * self.m[i]
                + (1.0 - self.beta1)
                * self.grads[i]
            )


            self.v[i] = (
                self.beta2 * self.v[i]
                + (1.0 - self.beta2)
                * (self.grads[i] ** 2)
            )


            m_hat = (
                self.m[i]
                / (1.0 - self.beta1 ** self.t)
            )


            v_hat = (
                self.v[i]
                / (1.0 - self.beta2 ** self.t)
            )


            self.params[i] -= (
                self.lr
                * m_hat
                / (np.sqrt(v_hat) + self.eps)
            )
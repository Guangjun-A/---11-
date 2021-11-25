import paddle

class FocalLoss(paddle.nn.Layer):
    def __init__(self, alpha_t=None, gamma=0):
        """
        :param alpha_t: A list of weights for each class
        :param gamma:
        """
        super().__init__()
        self.alpha_t = paddle.to_tensor(alpha_t) if alpha_t else None
        self.gamma = gamma

    def forward(self, outputs, targets):
        if self.alpha_t is None and self.gamma == 0:
            focal_loss = paddle.nn.functional.cross_entropy(outputs, targets, soft_label=True)

        elif self.alpha_t is not None and self.gamma == 0:
            # if self.alpha_t.device != outputs.device:
            #     self.alpha_t = self.alpha_t.to(outputs)
            focal_loss = paddle.nn.functional.cross_entropy(outputs, targets,
                                                           weight=self.alpha_t, soft_label=True)

        elif self.alpha_t is None and self.gamma != 0:
            ce_loss = paddle.nn.functional.cross_entropy(outputs, targets, reduction='none', soft_label=True)
            p_t = paddle.exp(-ce_loss)
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()

        elif self.alpha_t is not None and self.gamma != 0:
            # if self.alpha_t.device != outputs.device:
            #     self.alpha_t = self.alpha_t.to(outputs)
            ce_loss = paddle.nn.functional.cross_entropy(outputs, targets, reduction='none', soft_label=True)
            p_t = paddle.exp(-ce_loss)
            ce_loss = paddle.nn.functional.cross_entropy(outputs, targets,
                                                        weight=self.alpha_t, reduction='none', soft_label=True)
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()  # mean over the batch

        return focal_loss

class FocalLoss2(paddle.nn.Layer):
    def __init__(self, alpha_t=None, gamma=0):
        """
        :param alpha_t: A list of weights for each class
        :param gamma:
        """
        super().__init__()
        self.alpha_t = paddle.to_tensor(alpha_t) if alpha_t else None
        self.gamma = gamma

    def forward(self, outputs, targets):
        if self.alpha_t is None and self.gamma == 0:
            focal_loss = paddle.nn.functional.cross_entropy(outputs, targets)

        elif self.alpha_t is not None and self.gamma == 0:
            # if self.alpha_t.device != outputs.device:
            #     self.alpha_t = self.alpha_t.to(outputs)
            focal_loss = paddle.nn.functional.cross_entropy(outputs, targets,
                                                           weight=self.alpha_t)

        elif self.alpha_t is None and self.gamma != 0:
            ce_loss = paddle.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = paddle.exp(-ce_loss)
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()

        elif self.alpha_t is not None and self.gamma != 0:
            # if self.alpha_t.device != outputs.device:
            #     self.alpha_t = self.alpha_t.to(outputs)
            ce_loss = paddle.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = paddle.exp(-ce_loss)
            ce_loss = paddle.nn.functional.cross_entropy(outputs, targets,
                                                        weight=self.alpha_t, reduction='none')
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()  # mean over the batch

        return focal_loss

if __name__ == '__main__':
    outputs = paddle.to_tensor([[2., 1.],
                            [2.5, 1.]])
    # outputs = paddle.randn((2, 30))
    y = paddle.nn.functional.one_hot(paddle.to_tensor([0, 1]), num_classes=2)
    print(y)
    targets = paddle.nn.functional.label_smooth(y)
    # print(paddle.nn.functional.softmax(outputs, axis=1))

    fl= paddle.nn.CrossEntropyLoss(soft_label=True)#FocalLoss([0.25 for _ in range(2)], 2)

    print(fl(outputs, targets))
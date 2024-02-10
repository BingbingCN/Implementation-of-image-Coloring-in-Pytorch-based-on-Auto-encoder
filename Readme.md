# Implementation of image Coloring in Pytorch based on Auto encoder

### Dataset

This project uses the cat and dog data set to implement, the deployment path can refer to the figure below.[https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset)

![Untitled](Implementation%20of%20image%20Coloring%20in%20Pytorch%20based%20%2065450cf1a78742b29966c1c0cef364cc/Untitled.png)

### Model

Encoder

```jsx
class Encoder(nn.Module):
    def __init__(self, do_bn=False):
        super().__init__()
        self.block1 = self.inner_block(3, 32)
        self.block2 = self.inner_block(32, 64)
        self.block3 = self.inner_block(64, 128)
        self.block4 = self.inner_block(128, 256)
        self.block5 = self.inner_block(256, 384)
        self.grayscale = transforms.Grayscale(3)

    def inner_block(self, in_c, out_c, maxpool=2):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # 3, 160, 160
        x=self.grayscale(x)
        h1 = self.block1(x)  # 32, 80, 80
        h2 = self.block2(h1)  # 64, 40, 40
        h3 = self.block3(h2)  # 128, 20, 20
        h4 = self.block4(h3)  # 256, 10, 10
        h5 = self.block5(h4)  # 384, 5, 5

        return [h1, h2, h3, h4, h5]
```

Decoder

```jsx
class Decoder(nn.Module):

    def __init__(self, do_bn):
        super().__init__()
        self.inner1 = self.inner_block(384, 256)
        self.inner2 = self.inner_block(256, 128)
        self.inner3 = self.inner_block(128, 64)
        self.inner4 = self.inner_block(64, 32)
        self.inner5 = self.inner_block(32, 3, out=True)

        self.cb1 = self.conv_block(512, 256)
        self.cb2 = self.conv_block(256, 128)
        self.cb3 = self.conv_block(128, 64)
        self.cb4 = self.conv_block(64, 32)

    def inner_block(self, in_c, out_c, out=False):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU() if not out else nn.Sigmoid(),
            nn.Dropout(0.2) if not out else nn.Identity(),
        )

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, h):
        # 384, 5, 5
        x = h[-1]
        x = self.inner1(x)  # 256, 10, 10

        x = torch.concat([x, h[-2]], dim=1)
        x = self.cb1(x)
        x = self.inner2(x)  # 128, 20, 20

        x = torch.concat([x, h[-3]], dim=1)
        x = self.cb2(x)
        x = self.inner3(x)  # 64, 40, 40

        x = torch.concat([x, h[-4]], dim=1)
        x = self.cb3(x)
        x = self.inner4(x)  # 32, 80, 80

        x = torch.concat([x, h[-5]], dim=1)
        x = self.cb4(x)
        x = self.inner5(x)  # 3, 160, 160

        return x
```

### Result

After 70 epochs of execution on the RTX3060 12G, the training loss and test loss level off

![Untitled](Implementation%20of%20image%20Coloring%20in%20Pytorch%20based%20%2065450cf1a78742b29966c1c0cef364cc/Untitled%201.png)

### Prediction

![Untitled](Implementation%20of%20image%20Coloring%20in%20Pytorch%20based%20%2065450cf1a78742b29966c1c0cef364cc/Untitled%202.png)

### More

Email📧：cq9067@gmail.com
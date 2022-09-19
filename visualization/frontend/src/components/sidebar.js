import React from "react";

export default () => {
  const onDragStart = (event, nodeName, subpm) => {
    event.dataTransfer.setData("application/reactflow", nodeName);
    event.dataTransfer.setData("subparameters", subpm);
    event.dataTransfer.effectAllowed = "move";
  };
  return (
    <aside>
      <div className="category">Convolution Layer</div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "Conv2d", "'in_channels': 3 \n 'out_channels': 64 \n 'kernel_size': (3, 3) \n 'stride': (1, 1) \n 'padding': (1, 1)")}
        draggable
      >
        Conv2d
      </div>
      <div className="category">Pooling layers</div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "MaxPool2d", "'kernel_size': (2, 2) \n 'stride': (2, 2) \n 'padding': (0, 0) \n 'dilation': 1 \n 'return_indices': False \n 'ceil_mode': False")}
        draggable
      >
        MaxPool2d
      </div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "AvgPool2d", "'kernel_size': (2, 2) \n 'stride': (2, 2) \n 'padding': (0, 0)")}
        draggable
      >
        AvgPool2d
      </div>
      <div
        className="dndnode"
        onDragStart={(event) =>
          onDragStart(event, "AdaptiveAvgPool2d (ResNet)","'output_size': (1, 1)")
        }
        draggable
      >
        AdaptiveAvgPool2d (ResNet)
      </div>
      <div className="category">Padding Layers</div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "ZeroPad2d","'padding': 1")}
        draggable
      >
        ZeroPad2d
      </div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "ConstantPad2d","'padding': 2 \n 'value': 3.5")}
        draggable
      >
        ConstantPad2d
      </div>
      <div className="category">Activations</div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "ReLU", "'inplace': False")}
        draggable
      >
        ReLU
      </div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "ReLU6", "'inplace': False")}
        draggable
      >
        ReLU6
      </div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "Sigmoid","'dim': 1")}
        draggable
      >
        Sigmoid
      </div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "LeakyReLU", "'negative_slope': 0.01 \n 'inplace': False")}
        draggable
      >
        LeakyReLU
      </div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "Tanh")}
        draggable
      >
        Tanh
      </div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "Softmax", "'dim': 0")}
        draggable
      >
        Softmax
      </div>
      <div className="category">Normalization Layers</div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "BatchNorm2d", "'num_features': 512")}
        draggable
      >
        BatchNorm2d
      </div>
      <div className="category">Linear Layers</div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "Linear", "'in_features': 1 \n 'out_features': 1 \n 'bias': False")}
        draggable
      >
        Linear
      </div>
      <div className="category">Dropout Layers</div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "Dropout", "'p': 0.5 \n 'inplace': False")}
        draggable
      >
        Dropout
      </div>
      <div className="category">Loss Functions</div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "BCELoss","'weight': None \n 'size_average': True \n 'reduce': False \n 'reduction': Mean")}
        draggable
      >
        BCELoss
      </div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "CrossEntropyLoss","'weight': None \n 'size_average': True \n 'ignore_index': None \n 'reduce': True \n 'reduction': Mean \n 'label_smoothing': 0.0")}
        draggable
      >
        CrossEntropyLoss
      </div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "MSELoss","'size_average': True \n 'reduce': True \n 'reduction': Mean")}
        draggable
      >
        MSELoss
      </div>
      <div className="category">Utilities</div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "Flatten","'start dim': 1 \n 'end dim': -1")}
        draggable
      >
        Flatten
      </div>
      <div className="category">Vision Layers</div>
      <div
        className="dndnode"
        onDragStart={(event) => onDragStart(event, "Upsample","'size': None \n 'scale factor': None \n 'mode': Nearest \n 'align corners': None \n 'recompute scale factor': None \n")}
        draggable
      >
        Upsample
      </div>
    </aside>
  );
};
import React from "react";
import NodeColorProp from "../../NodeColor";

const layerToggle = () => {
  const onDragStart = (event, nodeName,  nodeColor,subpm) => {
    event.dataTransfer.setData("application/reactflow", nodeName);
    event.dataTransfer.setData("subparameters", subpm);
    event.dataTransfer.setData('colorNode',nodeColor);
    event.dataTransfer.effectAllowed = "move";
  };
  return (
      <div className="LayerToggle">
          <h2 className="Layer">Layer</h2>
    <aside>
        <details className="categoryYolo">
          <summary className="layerName">Yolo Block</summary>
            <ul>
              <li>
                <div
                  className="dndnode"
                  onDragStart={(event) => onDragStart(event, "Concat", `${NodeColorProp.Yolo}`, "'dim': 1")}
                  draggable
                >
                  Concat
                </div>
              </li>
            </ul>
        </details>
        <details className="categoryResidual">
          <summary className="layerName">Residual Block</summary>
              <ul>
                  <li>
                      <div
                        className="dndnode"
                        onDragStart={(event) => onDragStart(event, "BasicBlock", `${NodeColorProp.Residual}`,"'inplanes': 1 \n 'planes': 1 \n 'stride': 1 \n 'downsample': False \n 'groups': 1 \n 'base_width': 64 \n 'dilation': 1 \n 'norm_layer': None \n")}
                        draggable
                      >
                          BasicBlock
                      </div>
                  </li>
                  <li>
                      <div
                        className="dndnode"
                        onDragStart={(event) => onDragStart(event, "Bottleneck", `${NodeColorProp.Residual}`,"'inplanes': 1 \n 'planes': 1 \n 'stride': 1 \n 'downsample': False \n 'groups': 1 \n 'base_width': 64 \n 'dilation': 1 \n 'norm_layer': None \n")}
                        draggable
                      >
                          Bottleneck
                      </div>
                  </li>
              </ul>
        </details>
        <details className="categoryConv">
          <summary className="layerName">Convolution Layer</summary>
          <ul>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "Conv2d", `${NodeColorProp.Conv}`,"'in_channels': 3 \n 'out_channels': 64 \n 'kernel_size': (3, 3) \n 'stride': (1, 1) \n 'padding': (1, 1) \n 'bias': True")}
                    draggable
                  >
                      Conv2d
                  </div>
              </li>
          </ul>
      </details>
      <details className="categoryPool">
          <summary>Pooling Layers</summary>
          <ul>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "MaxPool2d", `${NodeColorProp.Pooling}`,"'kernel_size': (2, 2) \n 'stride': (2, 2) \n 'padding': (0, 0) \n 'dilation': 1 \n 'return_indices': False \n 'ceil_mode': False")}
                    draggable
                  >
                    MaxPool2d
                  </div>
              </li>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "AvgPool2d", `${NodeColorProp.Pooling}`,"'kernel_size': (2, 2) \n 'stride': (2, 2) \n 'padding': (0, 0)")}
                    draggable
                  >
                    AvgPool2d
                  </div>
              </li>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) =>
                      onDragStart(event, "AdaptiveAvgPool2d",`${NodeColorProp.Pooling}`,"'output_size': (1, 1)")
                    }
                    draggable
                  >
                    AdaptiveAvgPool2d
                  </div>
              </li>
          </ul>
      </details>
      <details className="categoryPad">
          <summary>Padding Layers</summary>
          <ul>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "ZeroPad2d", `${NodeColorProp.Padding}`,"'padding': 1")}
                    draggable
                  >
                    ZeroPad2d
                  </div>
              </li>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "ConstantPad2d",`${NodeColorProp.Padding}`,"'padding': 2 \n 'value': 3.5")}
                    draggable
                  >
                    ConstantPad2d
                  </div>
              </li>
          </ul>
      </details>
      <details className="categoryActi">
          <summary>Activations</summary>
          <ul>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "ReLU", `${NodeColorProp.Activation}`,"'inplace': False")}
                    draggable
                  >
                    ReLU
                  </div>
              </li>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "ReLU6", `${NodeColorProp.Activation}`,"'inplace': False")}
                    draggable
                  >
                    ReLU6
                  </div>
              </li>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "Sigmoid",`${NodeColorProp.Activation}`,"'dim': 1")}
                    draggable
                  >
                    Sigmoid
                  </div>
              </li>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "LeakyReLU", `${NodeColorProp.Activation}`,"'negative_slope': 0.01 \n 'inplace': False")}
                    draggable
                  >
                    LeakyReLU
                  </div>
              </li>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "Tanh",`${NodeColorProp.Activation}`)}
                    draggable
                  >
                    Tanh
                  </div>
              </li>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "Softmax", `${NodeColorProp.Activation}`,"'dim': 0")}
                    draggable
                  >
                    Softmax
                  </div>
              </li>
          </ul>
      </details>
      <details className="categoryNorm">
          <summary>Normalization Layers</summary>
            <ul>
                <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "BatchNorm2d", `${NodeColorProp.Normalization}`,"'num_features': 512")}
                    draggable
                  >
                    BatchNorm2d
                  </div>
                </li>
            </ul>
      </details>
      <details className="categoryLinear">
          <summary>Linear Layers</summary>
            <ul>
                <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "Linear", `${NodeColorProp.Linear}`,"'in_features': 1 \n 'out_features': 1 \n 'bias': False")}
                    draggable
                  >
                    Linear
                  </div>
                </li>
            </ul>
      </details>
      <details className="categoryDrop">
          <summary>Dropout Layers</summary>
          <ul>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "Dropout", `${NodeColorProp.Dropout}`,"'p': 0.5 \n 'inplace': False")}
                    draggable
                  >
                    Dropout
                  </div>
              </li>
          </ul>
      </details>
      <details className="categoryLoss">
          <summary>Loss Functions</summary>
          <ul>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "BCELoss",`${NodeColorProp.Loss}`,"'weight': None \n 'size_average': True \n 'reduce': False \n 'reduction': Mean")}
                    draggable
                  >
                    BCELoss
                  </div>
              </li>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "CrossEntropyLoss",`${NodeColorProp.Loss}`,"'weight': None \n 'size_average': True \n 'ignore_index': None \n 'reduce': True \n 'reduction': Mean \n 'label_smoothing': 0.0")}
                    draggable
                  >
                    CrossEntropyLoss
                  </div>
              </li>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "MSELoss",`${NodeColorProp.Loss}`,"'size_average': True \n 'reduce': True \n 'reduction': Mean")}
                    draggable
                  >
                    MSELoss
                  </div>
              </li>
          </ul>
      </details>
      <details className="categoryUtil">
          <summary>Utilities</summary>
          <ul>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "Flatten",`${NodeColorProp.Utilities}`,"'start dim': 1 \n 'end dim': -1")}
                    draggable
                  >
                    Flatten
                  </div>
              </li>
          </ul>
      </details>
      <details className="categoryVision">
          <summary>Vision Layers</summary>
          <ul>
              <li>
                  <div
                    className="dndnode"
                    onDragStart={(event) => onDragStart(event, "Upsample",`${NodeColorProp.Vision}`,"'size': None \n 'scale factor': None \n 'mode': Nearest \n 'align corners': None \n 'recompute scale factor': None \n")}
                    draggable
                  >
                    Upsample
                  </div>
              </li>
          </ul>
      </details>
    </aside>
          </div>
  );
};

export default layerToggle;

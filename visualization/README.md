# [22.07.27] Readme.md

### <Base Components>

![%EB%A0%88%EC%9D%B4%EC%96%B4_22](https://user-images.githubusercontent.com/52227367/181422519-59d6b34d-7505-4e13-a8c0-c9474d00af44.jpg)

① **TOOLBOX** : lists the types of layers that can be employed.

② **OVERVIEW DIAGRAM** : shows an overall architecture of target model. 

③ **ARCHITECTURE EDITOR** : enables users to edit an architecture of target model.

④ **GENERATE BUTTON** : enables user to generate a pth file based on the manipulated architecture.

⑤ **MINI MAP** : shows a status of an architecture of target model.

### <Additional Components>

![%EB%AA%A8%EB%8B%AC222](https://user-images.githubusercontent.com/52227367/181422573-2f03e8d0-4088-4927-9a6f-fd9927e87c81.jpg)

① **PARAMETER EDITOR** : enables users to fine-tune the parameter values.

② **DEFAULT BUTTON** : enables users to load the default parameter values.

③ **SAVE BUTTON** : enables users to save the current parameter value and dismiss the **PARAMETER EDITOR**.

### [How to add a layer]

- Drag a layer from the **TOOLBOX** to the **ARCHITECTURE EDITOR**.
- Can change the position of the layer by drag-and-drop.

### [How to delete a layer]

- Click the layer and press backspace key to delete the layer.
- Note that the edges connected with the deleted layer are removed together.

### [How to add an edge]

- Connect the points at the top and bottom of two different layers.

### [How to delete an edge]

- Press backspace key to delete the edge between the layers.

### [How to modify parameters of layer]

- Double-click a layer

![%EC%88%98%EC%A0%95](https://user-images.githubusercontent.com/52227367/181422622-76ce26ce-ceaa-4977-9cc7-dd9b1a451efe.png)

- Click the value you want to update in the pop-up window and press the **SAVE BUTTON** to save the value.
- Click **DEFAULT BUTTON** to set the default values.

### [How to generate pth file]

- Click the **GENERATE BUTTON** to create a pth file.

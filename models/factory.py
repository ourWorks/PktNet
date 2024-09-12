from .ablation._model_PktNet import PktNet
from .ablation._model_BaseModel import BaseModel
from .ablation._model_BaseModel_FeatureExtraLayer import BaseModel_FeatureExtraLayer
from .ablation._model_BaseModel_PacketLayer import BaseModel_PacketLayer
from .mpsffa_model import MPSFFA
from monai.networks.nets import resnet10,resnet18,resnet34,resnet50,resnet101,resnet152,resnet200
from monai.networks.nets import EfficientNet
from monai.networks.nets import ViT

from monai.networks.nets import DenseNet

# 比较的模型
from models.two_classes.DAMIDL import DAMIDL

class ModelFactory:

    def __init__(self):
        print("Building the model...")

    def create_model(self, model_name, num_classes):

        EfficientNet_blocks = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25",
        ]

        print(f"传入工厂类的名字为：{model_name}")

        # proposed model
        if model_name == "PktNet":
            return PktNet(num_classes=num_classes)
        
        ############################################################ 消融实验用的模型
        elif model_name == "BaseModel":
            return BaseModel(num_classes=num_classes)
        elif model_name == "BaseModel_FeatureExtraLayer":
            return BaseModel_FeatureExtraLayer(num_classes=num_classes)
        elif model_name == "BaseModel_PacketLayer":
            return BaseModel_PacketLayer(num_classes=num_classes)

        ############################################################ 专用模型
        elif model_name == "MPSFFA":
            return MPSFFA(num_classes=num_classes)
        
        ############################################################ 通用模型
        elif model_name == "DenseNet":
            return DenseNet(spatial_dims=3,in_channels=1,out_channels=num_classes)
        elif model_name == "EfficientNet":
            return EfficientNet(blocks_args_str=EfficientNet_blocks,spatial_dims=3, in_channels=1, num_classes=num_classes)
        elif model_name == "ResNet18":
            return resnet18(n_input_channels=1, num_classes=num_classes)
        elif model_name == "ResNet34":
            return resnet34(n_input_channels=1, num_classes=num_classes)
        elif model_name == "ResNet50":
            return resnet50(n_input_channels=1, num_classes=num_classes)
        # elif model_name == "ViT":
        #     return ViT(in_channels=1, img_size=(96,96,96), patch_size=16, classification=True, num_classes=num_classes, spatial_dims=3, save_attn=True)
        else:
            print(f"##${model_name}###")
            raise ValueError("不支持的模型")
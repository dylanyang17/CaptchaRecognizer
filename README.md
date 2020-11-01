# CapthcaRecognizer

## 简介

一个简单的验证码识别器，正在开发中...

## 运行流程

* data_process.py 的 main 中有三部分：通过 captcha 生成图片存放到 Config.IMAGE_DIR 下，并处理成数据存放到 Config.DATA_DIR 目录下；
* train.py 的 main 中载入了数据并进行训练；
* predict.py 的 main 中利用 Config.final_net_path 找到并读入网络模型进行预测。

## 关于字体

为了生成相应字体，这里用的方式是截取真实验证码的合适图片，并存放到 `fonts/single` 下，利用 FontCreator 软件创建字体即可。
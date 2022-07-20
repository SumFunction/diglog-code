**本项目基于PaddleNLP修改的多技能对话**<br>
**学号**：Z127 **学生姓名**：连榕榕 **就读学校**：福建农林大学 **专业**：软件工程<br>
# 代码结构
```
├── best_checkpoint
│   ├── model_best #最佳模型
├── best_result
│   ├── predict.txt #最佳预测结果
├── dataset #数据集存放处
│   ├── train.txt
│   ├── dev.txt 
│   ├── test.txt 
├── args.py #启动配置文件
├── train.py #训练脚本
├── infer.py #推理脚本
└── tools
    ├── convert_data_to_numberical.py 数据集序列化脚本
    ├── data_helper.py
    ├── data_process
    │   ├── __init__.py
    │   ├── convert_id.py
    │   └── to_sample_for_data_source.py
    └── input_args.py
```
# 项目启动
遵循下列步骤一步步完成即可实现项目顺利启动。
## Step1:解压数据集
```
#解压需要的数据集
unzip /home/aistudio/data/data157611/Dialog_train.zip -d /home/aistudio/data/data157611
unzip /home/aistudio/data/data157611/Dialog_dev.zip -d /home/aistudio/data/data157611
unzip /home/aistudio/data/data157611/Dialog_testA.zip -d /home/aistudio/data/data157611
unzip /home/aistudio/data/data157611/Dialog_testB.zip -d /home/aistudio/data/data157611
```
## Step2:解压代码文件
```
unzip work/dialog-code.zip
```
## Step3:转换数据集为规定格式
首先进入工作目录,使用如下脚本命令，将数据集进行转换
```
cd work/dialog-code  
python tools/convert_data_to_numerical.py tools/spm.model
```
> 在data_process_list中可以更改生成数据集的样本数量.
为节约生成时间，这里设置生成为2000条。可以去对应代码文件里修改生成训练集的大小![](https://ai-studio-static-online.cdn.bcebos.com/d316adbe4d6847ea85a7a0802c0c5f9a48d729ba32464a8aacfb252664da7743)



## Step4:启动训练脚本
```
python train.py \
--model_name_or_path=./best_checkpoint/model_best \
--train_data_path=./dataset/train.txt \
--valid_data_path=./dataset/dev.txt \
--save_dir=./checkpoints \
--logging_steps=100 \
--save_steps=1000 \
--seed=2021 \
--epochs=30 \
--batch_size=8192 \
--lr=1e-6 \
--weight_decay=0.0005 \
--sort_pool_size=65536 \
--device=gpu
```
> 默认为单卡训练，如果需要多卡训练则加命令：-m paddle.distributed.launch --gpus 0,1,2,3
## Step5：启动推理脚本
```
python infer.py \
--model_name_or_path=./best_checkpoint/model_best \
--test_data_path=./dataset/test.txt \
--output_path=./predict.txt \
--logging_steps=500 \
--seed=2021 \
--batch_size=4 \
--min_dec_len=1 \
--max_dec_len=64 \
--num_samples=20 \
--decode_strategy=sampling \
--top_k=5 \
--top_p=0.7 \
--device=gpu
```
# 基于baseline的改进过程
> 由于比赛为封榜机制，不能看到具体分数，具体改进过程只能依靠排名是否有上升来观察。
* **跑通baseline**<br>
  > 1.基于PaddleNLP版本的baseline<br>
  2.使用预训练模型'unified_transformer-12L-cn'<br>
  3.训练参数配置：lr=1e-5，warmup_steps=4000，AdamW优化器
  
  **最终效果**：best_ppl为56.13
 
 * **调整预训练模型，更改训练策略**
 > 1.使用预训练模型'unified_transformer-12L-cn-luge'<br>
 > 2.切换为多卡训练<br>
 > 3.训练参数配置：lr=4e-5，采用余弦退火式学习率下降，使用Momentum优化器进一步微调，训练10个epoch<br>
 > 4.训练集使用三部分数据（百度自建数据）做训练，展开后，共计5w条数据
 
   **最终效果**：best_ppl为：15.722
 * **加大训练epoch，扩充数据集**
 > 1.训练epoch调整为40，其余参数配置不变<br>
 > 2.使用比赛提供的全部数据集（共计采样200w条）
 
   **最终效果**：best_ppl为：14.271
* **调整解码策略**
> 使用beam_search进行预测解码，具体参数设置为num_samples=10，num_beams=10。

  **最终效果**：排名无变化
> 将解码策略调整为sampling，设置top-k,top-p同时进行。其中top-k=5,top-p=0.7，num_samples=20

  **最终效果**：排名上升3名
* **重构训练数据**
> 将验证集训练丢入进行训练
  
  **最终效果**：还未测试(来不及测试了)

# 个人总结
1. 通过本项目，掌握了基本NLP任务的处理流程。熟悉了常见的NLP对话任务中的模型。
2. 由于比赛时间仅有一周需要完成两个项目，故本项目待优化的地方还有很多，未来将会从以下几个方面对项目做进一步优化：
* 尝试使用PLATO-2模型（该模型在比赛中未找到中文版本的预训练模型，故未使用，今后若找到将会切换为此模型）。
* 模型结构还有很多优化的地方，例如goal细化表征，知识细化表征，知识模板化，copy机制等等。
* 尝试自定义的解码策略。


# 基于运营商大数据的栅格时序图预测

## 文件分布
1. baseline-for-GAT-and-BiLSTM-main:助教的代码
2. data:存储数据文件
3. prediction:存储预测文件
4. submitCSV:存储提交的CSV，按照日期和次数来作为文件名
5. test_npdata:测试集数据的numpy文件，可以直接读取
6. train_npdata:训练集数据的numpy文件，可以直接读取
7. total_test_npdata:将训练集和测试集合并后的numpy文件，用于最终的测试，可以直接读取
8. 代码直接放在了文件目录下(分为模型代码、主程序代码和数据处理代码三部分)
# ZalAI接口文档

[TOC]


ZalAI，分为ZalAIGui和ZalAIBack两部分，都可以独立运行，ZalAIBack为ZalAIGui程序提供神经网络训练服务。
ZalAI接口目前2020/02/10包括
1. ZalAIGui通过命令行接口调用ZalAIBack（命令行输入）
2. ZalAIBack通过管道输出状态数据到ZalAIGui（命令行输出）
3. ZalAIBack输出配置文件/缓存文件/输出文件到ZalAIGui（文件输入/文件输出）
4. 其他相关内容

1和2的部分必须按照约定完成。
ZalAIGui通过命令行调用训练脚本程序ZalAIBack，并传递参数。粗粒度参数通过命令行传递，细粒度参数通过保存成json文件传给训练脚本，训练程序输出训练状态参数，通过管道传递到ZalAIGui。
文档只对命令行调用命令部分进行约定，具体命令行的具体实现方式及具体解析不做规范。
文档对输出打印部分进行了详细约定，使用指定函数输出指定状态。
相关文件配置，包括json配置，csv输入，说明文档和ui文件。文档推荐使用json配置，并不具体指定文件的内部参数如何命名和使用。ui文件需要提供朴素草图，控件与命令行参数对应。

##  命令行接口
程序的命令行接口包含以下内容
* 程序包含训练train/预测predict/单元测试unit/导出export等子命令(必须)
* 支持指定数据集的文件目录或文件 (必须)
* 支持配置模型文件保存路径或导入路径 (必须)
* 支持通过json文件导入配置 (可选项）
* 支持通过csv文件导入数据 (可选项)

路径参数要求支持中文路径，带空格路径

### 接口简介

重点在于命令行的命令**输入格式**，只要符合以下输入方式。
首先是位置参数，必填参数，字符串值为`train`,`predict`，`unittest`，`export`，主要用于区分训练和预测或其他函数。

1. 训练神经网络模型，关键字为`train`，必须实现
2. 神经网络模型提供预测，关键字为`predict`，必须实现
3. 神经网络模型用例单元测试，关键字为`unittest`，必须实现
4. 神经网络模型导出为目标平台二进制文件，关键字为`export`
5. 神经网络模型提供网络服务，关键字为`serve`
6. 测试神经网络模型性能，关键字为`test`
7. 提供路径，生成适配的数据样本(csv文件)，关键字为`gencsv`

其他参数为非位置参数，顺序可变。非位置参数包括输入输出路径参数，训练过程超参数，和其他配置。

#### 训练
通过位置参数train指定，格式为 `zalAi.exe train --input ... --output ... `
* `--input`指定后续输入参数为训练集目录或文件（必须实现/必填参数）
* `--valids`指定后续输入参数为验证集目录或文件（必须实现/可选参数）
* `--output`指定后续输入参数为模型输出文件路径 （必须实现）
* `--model-in`指定后续输入参数为模型输入文件路径，用于模型再训练。
* `--csv-input`指定csv输入文件路径 （必须实现）
* `--valid-split`指定从输入集中切分验证集, （可选实现）
* `--epochs`指定epchs, （可选实现）
* `--batch-size`指定batch_size, （可选实现）
* `--learn-rate` 指定学习率 （可选实现）


考虑到样本目录有多种方式，以上的训练集和验证集和测试集路径可以是文件夹或单文件或多文件，
`--input --valids --test`对不同算法（分类或拟合）有不同输入格式，基于训练任务和预测任务不同又有区别，输入格式包括以下几种：
* 输入文件夹路径，输入文件后缀名（如.png/.jpg/.txt等），文件夹下有多个子文件夹，子文件夹对应一个类别标签，子文件夹下有多个目标后缀文件
* 输入单个csv文件，以及同名的csv.json配置文件，csv.json配置文件指定输入输出信息。
* 输入单个文件，后缀名为.png/.jpg/.txt等
* 输入单个文件夹，输入文件后缀名（如.png,.jpg,.txt等），文件夹下有多个目标后缀文件

需要在命令行ini文件中强调输入`--input`是文件还是文件夹，输出`--output`是文件还是文件夹。

#### 预测
通过位置参数train指定，格式为 `zalAi.exe train --test ... --output ... `
* `--model-in`指定后续输入参数为模型输入文件路径，用于模型再训练。
* `--test`指定后续输入参数为预测输入路径 （必须实现）
* `--gpu-on`指定是否开启GPU，可以提供默认值True或False。（可选参数）

#### 单元测试
提供快速运行训练用例和预测用例，使用默认参数，避免配置参数的繁琐情况。内置all,predict,train的情况，其他测试用例可以自行添加。

``` python
def unittestWr(run,**kwargs):
    global test_path,train_path,model_file,config_path
    if run == "all" or  run == "train" :
        train4Wr( input=train_path,output=model_file,config=config_path,**kwargs )
    if run == "all" or  run == "predict" :
        predict4Wr(model_in=model_file,test_path=test_path ,**kwargs)
def main():
    parserT = subparsers.add_parser('unittest', help='sensor unittest')
    parserT.add_argument('--run','-r',default="all",choices=["all","train","predict"],help ="unittest to train or predict ")
    parserT.set_defaults(handle = unittestWr )
```
#### 生成数据快照
输入路径和可选配置，生成数据样本快照。

#### 导出
略
#### 服务
略
*开启服务之后，通过web服务，为zalAIGui提供网络服务。*
### 命令行示例 

#### 参考参数
以下是常见的命令行通用配置项
* `--gpu-on`指定是否开启GPU，可以提供默认值True或False。（可选实现）
* `--gpu-device`指定训练使用的Nvidia的GPU名称或编号,默认提供挑选策略,适用于用户电脑有多块Nvidia显卡的情况（可选实现）  
* `--host`指定tcp连接IP地址 （待添加）
* `--port`指定tcp连接端口号 （待添加）
* `--token` 指定程序的口令密钥 （待添加）
* `--log-level` 指定记录级别 （待添加）

#### 命令行demo

命令行的具体实现方式没有统一规范，使用argparse/optparse/fire/click都可以实现解析。
以下是一个命令行接口的设计demo,sensorMain.py文件的部分内容

``` python

class Cli(object):
    def __init__(self,prog_name= "sensor fit module",nn=Trainer):
        self.nn = nn()
        self.config = ZalConfig()
        self.prog_name = prog_name
        self.update_command_config()
    def setNetwork(self,nn):
        self.nn = nn
    def update_command_config(self):
        pass
    def run(self,cmd=None):
        fStdoutStatus(TrainProc.ON_ARGPARSE_START)
        if cmd is None:
            cmd = sys.argv
        self.config._command = cmd
        # fStdoutDict(dict(time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()),status = TrainProc.START),'/status')

        self.parent_parser = argparse.ArgumentParser(add_help=False)
        self.parent_parser.add_argument('--logging-level', '-l',
                            choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL', 'CRITICAL'],
                            default='INFO', action="store", help="the level of logging to use")
        self.parent_parser.add_argument('--logging-file', '-f', default="tmp_zalAI.log", dest="log_file", action="store",
                            help="logging file")
        self.parent_parser.add_argument('--logging-disable', '-lds', default=False, action="store_true", dest="log_disable",
                            help="use logging  or not")
        group = self.parent_parser.add_mutually_exclusive_group()
        group.add_argument('--gpu-on',dest='gpu-on', action='store_true',help ="flag to use gpu,default is without gpu ")
        group.add_argument('--gpu-off', dest='gpu-on',action='store_false',help ="flag to not use gpu")
        # parser.add_argument('--gpu-on',default = False,action="store_true",help ="flag to use gpu,default is without gpu ")

        parser = argparse.ArgumentParser(prog=self.prog_name,
            description='%s train or predict cmdline' % self.prog_name,
            usage='''%s <command> [<args>]

Available sub-commands:
   train                 Trains a model
   predict               Predicts using a pretrained model
   serve                 Serves a pretrained model
   unittest              Tests a pretrained model
   visualize             Visualizes experimental results
   gencsv                generate csv file from data file or sample follder 
'''%self.prog_name)

        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(cmd[1:2])
        if  args.command  in ["train","predict","unittest","gencsv","export"]:
            getattr(self, args.command)(cmd[2:])
        else:
            print('Unrecognized command')
            parser.print_help()
        # exit(1)

    def train(self,cmd=None):
        parser = argparse.ArgumentParser('train',parents=[self.parent_parser], description='%s train' % self.prog_name)
        parser.add_argument('--input','-i',default=self.config.input_path,action="store",required=True, dest="input",help="input file/folder")
        parser.add_argument('--valids', '-v', action="store", dest="valid_path", help="valids folder")
        parser.add_argument('--output','-o',default=self.config.model_file,action="store", dest="output",help="output file")
        parser.add_argument('--config','-c',default=self.config.cfg_path,action="store", dest="config")
        parser.add_argument('--csv-cfg', action="store", dest="csv_cfg")
        parser.add_argument('--valid-split',default=0.1, type = float,action="store", dest="valid_split")
        parser.add_argument('--test-split' ,default=0.1, type = float,action="store", dest="test_split")
        # parser.add_argument('--csv-input', action="store", dest="csv_input")
        # parser.add_argument('--model-name',  action="store", dest="model_name", help="output model_name")
        parser.add_argument('--model-in','-m',action="store", dest="model_in")
        parser.add_argument('--input-type','-it',action="store", dest="input_type",
            default=INPUT_TYPE.CSV,choices=INPUT_TYPE.ALL,help ="decide input is csv file or other file or directory ")
        parser.add_argument('--output-type',action="store", dest="output_type",
            default="file",choices=["file","dir"],help ="decide output is file or directory ")

        group = self.parent_parser.add_mutually_exclusive_group()
        group.add_argument('--datagen',dest='gen_on', action='store_true',help ="flag to use data generator,default is data all in ")
        group.add_argument('--dataall', dest='gen_on',action='store_false',help ="flag to not use data generator")
        group2 = self.parent_parser.add_mutually_exclusive_group()
        group2.add_argument('--use-temp',dest='temp_on', action='store_true',help ="flag to use temp data ,default is without temp ")
        group2.add_argument('--no-temp', dest='temp_on',action='store_false',help ="flag to not use temp data")

        args = parser.parse_args(cmd)
        setLogger(__name__, args.logging_level, args.log_disable)
        dct = vars(args)
        logger.info(dct)
        fStdoutStatus(TrainProc.ON_ARGPARSE_END)
        self.nn.trainWrap(**dct)

    def predict(self,cmd=None):
        parser = argparse.ArgumentParser('predict',parents=[self.parent_parser], description='%s predict'%self.prog_name)
        parser.add_argument('--test','-t',default=self.config.test_path,action="store", dest="test_path")
        parser.add_argument('--model-in','-m',default=self.config.model_in_path,action="store", dest="model_in")
        parser.add_argument('--config','-c',default=self.config.cfg_path,action="store", dest="config")
        # parser.add_argument('--gpu-on',default = False,action="store_true",help ="flag to use gpu ")
        
        args = parser.parse_args(cmd)
        dct = vars(args)
        logger.info(dct)
        fStdoutStatus(TrainProc.ON_ARGPARSE_END)
        self.nn.predictWrap(**dct)

    def unittest(self,cmd=None):
        parser = argparse.ArgumentParser('unittest',parents=[self.parent_parser], description='%s unittest'%self.prog_name)
        parser.add_argument('--run','-r',default="all",choices=["all","train","predict"],help ="unittest to train or predict ")
        parser.add_argument('--test_file',default=self.config.test_path)
        parser.add_argument('--train_path',default=self.config.train_path)
        parser.add_argument('--model_file',default=self.config.model_file)
        parser.add_argument('--config_path',default=self.config.cfg_path)
        # parser.set_defaults(handle = unittestWrap )
        args = parser.parse_args(cmd)        
        dct = vars(args)
        logger.info(dct)
        fStdoutStatus(TrainProc.ON_ARGPARSE_END)
        self.nn.unittestWrap(**dct)
    def gencsv(self,cmd=None):
        parser = argparse.ArgumentParser('gencsv', description='%s gencsv'%self.prog_name)
        parser.add_argument('--input','-i',default=self.config.input_path)
        parser.add_argument('--csv-cfg', action="store",default=None, dest="csv_cfg")
        args = parser.parse_args(cmd)        
        dct = vars(args)
        logger.info(dct)
        fStdoutStatus(TrainProc.ON_ARGPARSE_END)
        self.nn.gencsvWrap(**dct)
        
    def export(self,cmd=None):
        parser = argparse.ArgumentParser('export',parents=[self.parent_parser], description='%s export'%self.prog_name)
        parser.add_argument('--output','-o',default=self.config.export_path,action="store", dest="export_path")
        parser.add_argument('--model-in','-m',default=self.config.model_in_path,action="store", dest="model_in")
        parser.add_argument('--config','-c',default=self.config.cfg_path,action="store", dest="config")       
        
        args = parser.parse_args(cmd)
        dct = vars(args)
        logger.info(dct)
        fStdoutStatus(TrainProc.ON_ARGPARSE_END)
        self.nn.exportWrap(**dct)
    def visualize(self,cmd):
        pass
    def serve(self,cmd):
        pass   
```

##  管道输出

训练过程中，穿插状态输出代码，打印当前训练阶段和训练过程数据信息。
状态输出使用json格式，使用可以用json.dumps打印的字典或列表的嵌套结构，支持None/浮点/整数。
fStdoutPack打印字符串。
fStdoutDict打印可json表示的字典或列表。
fStdoutDict第二个参数表示当前字典或列表嵌套结构的变量分发目标地址
**注意：不能直接使用json导出numpy的numpy.array等格式，需要自行定义json导出方式**
为了节约位数，浮点数至多使用单精度，瞬时训练参数推荐只保留4位精度。

fStdout函数原型定义如下所示
``` python
# fStdoutDict使用方法类似下方的伪函数
def fStdoutPack(x,host = '/status'):
    print(x,host)
def fStdoutDict(x,host="/status",**entries):
    fStdoutPack( json.dumps(x),host)
```


### 状态量
输出变量有"status","data","prediction","performance","config"
对应的分发地址(host)为"/status","/data","/prediction","/performance","/config"

* "status"变量描述训练过程阶段信息，包括程序开始，加载数据，开始训练等等。
* "data"描述**训练过程**的batch,epoch等瞬时参数,包括训练集的acc,loss，验证集的部分用valid_acc,valid_loss,拟合增加mse，mae，mape,msle等指标
* "config"描述网络开始训练前的配置参数，但这里只需要重要超参数，包括batch_size, batches,epochs,learn_rate,train_samples,valid_samples。
* "performance" 描述训练**网络整体**的性能，主要包括 loss和acc，还有召回率recall，精确率precision，AUC，ROC,验证集的指标增加valid_前缀，例如valid_loss,valid_acc，拟合增加mae,msle等指标。
* "prediction"描述预测返回结果，包括prob，label,predict,boxes。
#### status
status已经定义了enumeration类型的TrainProc常量。
``` python
from enum import Enum
# 描述训练的阶段状态
class TrainProc(Enum):
    ON_START = "on_start"  # 程序调用开始，与ON_EXIT成对
    ON_ARGPARS_START = "on_argparse_start"  # python执行命令行解析之前
    ON_ARGPARS_END = "on_argparse_end"  # python执行命令行解析结束

    ON_TRAIN_START = "on_train_start"   # 开始训练 
    ON_TRAIN_END = "on_train_end"   # 结束训练 
    ON_READYDATA_START = "on_readydata_start" # 训练之前的数据准备网络准备开始
    ON_READYDATA_END = "on_readydata_end" # 结束准备
    ON_FIT_START = "on_fit_start" # 拟合开始
    ON_FIT_END = "on_fit_end" # 拟合准备

    ON_EVALUTE_START = "on_evalute_start"  # 开始评估模型性能
    ON_EVALUTE_END = "on_evalute_end"  # 结束评估模型性能
    ON_PREDICT_START = "on_predict_start" # 开始预测
    ON_PREDICT_END = "on_predict_end" # 结束预测
    ON_EXPORT_START = "on_export_start" # 开始导出
    ON_EXPORT_END = "on_export_end" # 结束导出
    ON_GENCSV_START = "on_gencsv_start" # 开始生成csv
    ON_GENCSV_END = "on_gencsv_end" # 结束生成csv
    ON_TEST_START = "on_test_start" # 开始test
    ON_TEST_END = "on_test_end" # 结束test
    ON_EXIT = "on_exit" # python退出程序之前
```


#### config
"config"描述网络配置参数，数据加载完成并冻结，开始训练之前的重要参数。其中`epochs`，`batches`，`batch_size`与绘图相关，必须提供
`epochs` 指定所有epoch次数，epoch指单次/第N次的epoch，
`batches` 指定所有batch次数，batch指单次/第N次的batch，
`batch_size` 指定单个batch大小
`train_samples`指所有训练集样本数目
`valid_samples` 指所有验证集样本数目
`learn_rate`: 初始学习率，

#### data
"data"描述**训练过程**的时间状态信息，可以绘成时序图。
时间度量是batch,epoch等瞬时参数。
状态度量包括训练集的acc,loss，验证集的部分用valid_acc,valid_loss,拟合增加mse，mae，mape,msle等指标
`batch`: 训练过程的第N次的batch
`epoch`: 训练过程的第N次的epoch  (必须提供)
`acc`: 训练样本的准确率。        (分类问题必须提供)
`loss`: 训练样本的损失值。       (必须提供)
`valid_acc`: 验证集样本的准确率
`valid_loss`: 验证集样本的损失值
`mse`: 训练样本的均方误差。


分类问题需要提供acc,其他问题如果没有acc这个指标，就不用输出acc。
准确率的格式为"%f" % (0.953) 。不要额外添加百分号，也不用放大100倍。
``` python
from eDataProtocol import fStdoutDict,fStdout,TrainProc
## Displayed the loss values and accuracy in training
class ShowLossAndAccuracy(keras.callbacks.Callback):
    def __init__(self):
        super(ShowLossAndAccuracy, self).__init__()
        self.batch_count = 0
        self.epoch_count = 0
    def on_batch_end(self, batch, logs={}):
        lst = {"batch":int(self.batch_count),"epoch":int(self.epoch_count), 
                 "loss":round(float(logs.get('loss')),3),"acc": round(float(logs.get('acc')),3)  }
        fStdoutDict( {"data": lst }) 
        self.batch_count += 1
    def on_epoch_end(self, epoch, logs={}): 
        self.epoch_count += 1
        self.batch_count = 0
```


#### prediction
`predict` 指定预测输出，分类问题是预测值索引(int)，拟合问题是预测值。不同的问题有不同输出，这部分需要具体协商，详细补充协议。
`label` 指定预测的类别标签，str类型。
`prob`指定预测输出的置信度或概率，float类型
`boxes` 指定检测对象的区域坐标，[int,int,int,int]类型
prediction可以通过直接打印输出：`fStdoutDict({"prediction":[ {"lable':"cat","boxes":[3,4,5,6]  },{"label":"dog","boxes":[3,4,5,6],"prob":0.99 }] })`
或者以文件输出，此时只需要打印文件名即可：`fStdoutDict( {"prediction": {"attachment": predict_file }},'/prediction') `


#### performance
描述训练**网络整体**的性能，主要包括 loss和acc，还有召回率recall，精确率precision，AUC，ROC,验证集的指标增加valid_前缀，例如valid_loss,valid_acc，拟合增加mae,msle等指标。
`ROC` 备选项，分类问题返回ROC的横坐标和纵坐标，例如`{"ROC":{"x":[1,2,3],"y":[4,5,6]}}`
召回率`recall`，精确率`precision`，`f1-score`等等

`{"performance": {"loss": 0.133, "acc": 0.951, "loss_valid": 0.13, "acc_valid": 0.954,"ROC":{"x":[1,2,3],"y":[4,5,6]}}}`
考虑到分类问题和拟合问题稍有区别，部分变量不通用。分类问题需要提供acc,其他问题没有acc，则无需提供acc。

## demo
``` python

class Trainer(object):
    def __init__(self):
        self.model = Model()
        self.dataLoader = CsvLoader()
        self.train_cfg = {
            "train_params":{
                "batch_size":5,
                "epochs":50,
                "verbose":0
            },
            "compile_params":{
                "loss":'mean_squared_error',
                "optimizer":'adam',
                'metrics':[]
                # "metrics":['mean_absolute_percentage_error','mean_absolute_error'],
            }             
        }
    def setModel(self,model):
        self.model = model
    def setDataLoader(self,dataLoader):
        self.dataLoader = dataLoader
    def _train(self,datDict): # todo
        datDict={ "x_train":x_train,"x_valid":x_valid,"x_test":x_test,
                    "y_train":y_train,"y_valid":y_valid,"y_test":y_test }

    def train_csv(self,output="boston_price.hdf5",csv_path=None,csv_json_path=None,valid_path=None,model_json_cfg = None,**kwargs):
        train_params=self.train_cfg["train_params"]
        trains = self.train_cfg["compile_params"]
        output_dir = os.path.splitext(output)[0]
        (x_train,y_train,x_valid,y_valid,x_test,y_test)=self.dataLoader.load_data_full(csv_path,csv_json_path,
                vocab_path=output_dir+'.vocab.json',valid_path=valid_path)
        
        ########################################
        net_cfg = {"in_shape": x_train.shape[1] ,"out_shape":y_train.shape[1],"mid_shape":8}
        
        if model_json_cfg is None:
            self.model.gen_model(**net_cfg)
            model_json_cfg  = output_dir+'.network.json' 
            self.model.save_model_to_json( model_json_cfg )
        else:
            self.model.load_model_from_json( model_json_cfg )
        self.model.compile_model(**trains)      
        self.model._model.summary()        

        train_params_freeze= copy.deepcopy(train_params)
        train_params_freeze.update({"batches": x_train.shape[0] // train_params_freeze["batch_size"]})
        fStdoutDict(dict( config = train_params_freeze ),"/config")
        # train_params.update({"class_weight":{0: 50, 1: 1} }) # 
        # class_weight = cw
        fStdoutStatus(TrainProc.ON_READYDATA_END)
        fStdoutStatus(TrainProc.ON_FIT_START)
        history = self.model.train( x_train,y_train,validation_data=[x_valid, y_valid],shuffle=True,
                    callbacks=[
                        ShowLossAndAccuracy(metrics=trains["metrics"]),
                        EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto'),
                        CSVLogger(output_dir+".training.csv", separator=',', append=False),
                        ModelCheckpoint(output,monitor='val_loss',
                        verbose=1,save_best_only=True, period=1)
                    ], **train_params)
        fStdoutStatus(TrainProc.ON_FIT_END)
        if x_test is not None:
            self.evaluateWrap(x_test,y_test)
    @fStdoutStatusDecorator(TrainProc.ON_TRAIN_START,TrainProc.ON_TRAIN_END) 
    def trainWrap(self,input=None,output="boston_price.hdf5",csv_json_path=None,valid_path=None,input_type=INPUT_TYPE.CSV,model_json_cfg = None,**kwargs):
        fStdoutStatus(TrainProc.ON_READYDATA_START)
        csv_path,csv_json_path = CsvLoader.gen_csv_by_type(input,input_type=input_type,csv_json_path=csv_json_path)
        if valid_path:
            valid_path,_ = CsvLoader.gen_csv_by_type(valid_path,input_type=input_type,csv_json_path=csv_json_path)

        logger.debug( (output,csv_path,csv_json_path,valid_path,model_json_cfg))
        self.train_csv(output,csv_path,csv_json_path,valid_path,model_json_cfg,**kwargs)
        
    @fStdoutStatusDecorator(TrainProc.ON_EVALUTE_START,TrainProc.ON_EVALUTE_END) 
    def evaluateWrap(self,x_test,y_test):
        perf = self.model.evaluate(x_test,y_test)
        logger.info("perf: %s" %(perf))
        fStdoutPack( json.dumps({"performance": perf },cls=NpEncoder),'/performance')  
        return perf

    @fStdoutStatusDecorator(TrainProc.ON_PREDICT_START,TrainProc.ON_PREDICT_END)
    def predictWrap(self,test_path ,model_in=None,**kwargs): 
        if model_in:
            self.model.load_model(model_in)
        vocab_file = os.path.splitext(model_in)[0]+'.vocab.json'
        self.dataLoader.load_vocab(vocab_file)
        x_test,_ = self.dataLoader.encode_data_from_csv(test_path)
        if self.model:      
            pred= self.model.predict(x_test)
            
            pred = self.dataLoader.decode_by_vocab(pred,self.dataLoader._collect_output() )
            
            logger.info( pred ) 
            # lst = [{"label":int(m),"prob":round(float(y[m]),3) }  for m in pred]
            # fStdoutDict( {"prediction": lst },'/prediction') 
            predict_file =  os.path.splitext(model_in)[0]+".predict.csv"
            df = pd.read_csv(test_path)
            if df.shape[0]==pred.shape[0]:
                pred.columns = ["predict_"+col for col in  pred.columns ]
                pred = pd.concat([df, pred],axis = 1)   
            pred.to_csv(predict_file,index=False)

            fStdoutDict( {"prediction": {"attachment": predict_file }},'/prediction') 
            # lst = [{"label":round(lb,3),"prob":round(float(1),3) }  for lb in pred[pred.columns[0]]]
            # fStdoutDict( {"prediction": lst },'/prediction') 
            return pred

    @fStdoutStatusDecorator(TrainProc.ON_UNITTEST_START,TrainProc.ON_UNITTEST_END)
    def unittestWrap(self,train_path,model_file,config_path,test_file,run= "all",**kwargs):
        if run == "all" or  run == "train" :
            self.trainWrap( input=train_path,output=model_file,config=config_path,**kwargs )
        if run == "all" or  run == "predict" :
            self.predictWrap(model_in=model_file,test_path=test_file ,**kwargs)

    @fStdoutStatusDecorator(TrainProc.ON_GENCSV_START,TrainProc.ON_GENCSV_END)
    def gencsvWrap(self,input=None,csv_json_path=None,input_type=INPUT_TYPE.CSV,**kwargs):
        csv_path,csv_json_path = CsvLoader.gen_csv_by_type(input,input_type=input_type,csv_json_path=csv_json_path,typeName=[".csv"])
        logger.info([csv_path,csv_json_path])
        return csv_path,csv_json_path
    @fStdoutStatusDecorator(TrainProc.ON_EXPORT_START,TrainProc.ON_EXPORT_END)
    def exportWrap(self,input=None,csv_json_path=None,input_type=INPUT_TYPE.CSV,**kwargs):
        pass
```
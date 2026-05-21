# Inference Frameworks

## Parts & Functions


framework概述：

server + clients结构：一个server提供推理服务端口，多个clients通过向server发起请求得到结果

接收请求 
｜
多个请求通过scheduler以及对应的调度策略组成batch
｜
batch prefill
|
batch decode
|
结果处理和返回

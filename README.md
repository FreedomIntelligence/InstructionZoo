# InstructionZoo

A collection of open-source Instruction-tuning dataset to train chat-based LLMs (ChatGPT,LLaMA,Alpaca).

This is an on-going project.


## Alpaca in different languages

| Dataset | Size | Language | Generation method |
|:---------| :---------:|:---------:|:---------|
| [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) |  52002 | EN | Self-instruct with human written 175 seed tasks using text-davinci-003 |
| [gururise/AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned) | 51713 | EN | A cleaned version of Stanford Alpaca Dataset, in order to solve issues like Hallucinations, Merged Instructions, Empty outputs, etc.|
| [carbonz0/alpaca-chinese-dataset](https://github.com/carbonz0/alpaca-chinese-dataset) | 20456 | CH | Translate Stanford Alpaca dataset into Chinese by machine, then self-instruct.|
| [hikarming/alpaca_chinese_dataset](https://github.com/hikariming/alpaca_chinese_dataset/tree/main/%E7%BF%BB%E8%AF%91%E5%90%8E%E7%9A%84%E4%B8%AD%E6%96%87%E6%95%B0%E6%8D%AE) | 19442 | CH | Translate Stanford Alpaca dataset into Chinese by ChatGPT, and check them by humans.|
| [ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/data) | 51458 | CH | Translate Stanford Alpaca dataset into Chinese by ChatGPT API, and discard some of them.|
| [A-baoYang/alpaca-7b-chinese](https://github.com/A-baoYang/alpaca-7b-chinese/blob/main/data/alpaca-zhTW.json) | 20465 | TC | Translate Stanford Alpaca dataset into traditional Chinese using OpenCC.|

## Instruction Dataset Collection

| Dataset | Size | Language | Domain | Generation method |
|:---------| :---------:|:---------:|:---------:|:---------|

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
| [A-baoYang/alpaca-7b-chinese/.../alpaca-zhTW.json](https://github.com/A-baoYang/alpaca-7b-chinese/blob/main/data/alpaca-zhTW.json) | 20465 | TC | Translate Stanford Alpaca dataset into traditional Chinese using OpenCC.|
| [A-baoYang/alpaca-7b-chinese/.../alpaca-en-zh.json](https://github.com/A-baoYang/alpaca-7b-chinese/blob/main/data/alpaca-en-zh.json) | 124469 | CH, EN | Combine the English instruction/input and traditional Chinese output by ChatGPT API ( gpt-3.5-turbo) .|
| [ntunlplab/traditional-chinese-alpaca/.../alpaca-tw_en_instruction.json](https://github.com/ntunlplab/traditional-chinese-alpaca/blob/main/data/alpaca-tw_en_instruction.json) | 52002 | CH, EN | A Traditional-Chinese version of the Alpaca dataset, whose instruction part is left as English. |
| [ntunlplab/traditional-chinese-alpaca/.../alpaca-tw_en-align.json](https://github.com/ntunlplab/traditional-chinese-alpaca/blob/main/data/alpaca-tw_en-align.json) | 52002 | CH, EN | An Traditional-Chinese version of the Alpaca dataset, where there are English and traditional Chinese versions of one single instruction. |
| [LC1332/Chinese-alpaca-lora](https://github.com/LC1332/Chinese-alpaca-lora/blob/main/data/trans_chinese_alpaca_data.json) | 51672 | CH | Translate Stanford Alpaca dataset into Chinese by ChatGPT API.|

## Instruction Dataset Collection

| Dataset | Size | Language | Domain | Generation method |
|:---------| :---------:|:---------:|:---------:|:---------|
| []() | | | | |
| [hikarming/alpaca_chinese_dataset](https://github.com/hikariming/alpaca_chinese_dataset/tree/main/%E5%85%B6%E4%BB%96%E4%B8%AD%E6%96%87%E9%97%AE%E9%A2%98%E8%A1%A5%E5%85%85) | 226 | CH | topic-specific |Generate Chinese instructions under various topics by humans, such as bussiness management, education, Romance of the Three Kingdoms, etc. |
| [sahil280114/codealpaca](https://github.com/sahil280114/codealpaca) | 20023 | EN | Code | Self-instuct with prompts to focus on code generation/edting/optimization tasks, using text-davinci-003. |
| [XueFuzhao/InstructionWild](https://github.com/XueFuzhao/InstructionWild) | 52191 \\ 479 seeds | CH, EN | | Collect 429 instructions from ChatGPT usage screenshots and release both English and Chinese versions, using text-davinci-003. |

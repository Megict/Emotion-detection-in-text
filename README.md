# Определние эмоций в тексте
Выпускная квалификационная работа бакалавра на тему "Определение эмоций персонажей в тексте посреством анализа диалогов с помощью машинного обучения".  
Основные составляющие программы:
- Алгоритм извлечения диалога из текста, который определяет персонажей в тексте и устанавливает, кто его произнес каждое высказывание.
- Детектор эмоций, который определяет эмоции, выраженные в каждом высказывании.

Точность извлечения диалога – **0.81**, точность определения эмоций – **0.79**.  

В алгоритме извлечения диалогов использовался синтаксичексий и морфологический парсер из библиотеки [_slovnet_](https://github.com/natasha/slovnet) и [_rupostagger_](https://github.com/Koziev/rupostagger).  
Для определения эмоций использовалась модель [_BERT_](https://huggingface.co/DeepPavlov/rubert-base-cased), предобученная на датасете [_cedr_](https://huggingface.co/datasets/cedr).  

### Emotion detection in text
Recognising emotions of сharacters in russian literary text by analysing dialogues.  
Main parts of the program are:  
- Dialogue extraction algorithm capable of detecting utterances pronounced by said character.  
- Emotion detection algorithm that can detect emotions present in an utterance.  

Accuracy of dialogue extraction is **0.81**, accuracy of emotion detection is **0.79**.  

These pretrained models were used:
- Syntax / morphological parsers [_slovnet_](https://github.com/natasha/slovnet) and [_rupostagger_](https://github.com/Koziev/rupostagger).
- [_RuBERT_](https://huggingface.co/DeepPavlov/rubert-base-cased) language model.

Emotion detector based on BERT type model was trained in two steps using [_cedr_](https://huggingface.co/datasets/cedr) dataset.

# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py
import string
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset, DatasetDict
import pandas as pd

import os
import json
import torch
import inspect
import tiktoken
import numpy as np
from tqdm import tqdm, trange
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from transformers.utils import cached_file

from llmtuner.data.template import get_template_and_fix_tokenizer
from llmtuner.eval.MMCBtemplate import get_eval_template
from llmtuner.extras.constants import CHOICES, SUBJECTS
from llmtuner.model import dispatch_model, get_eval_args, load_model_and_tokenizer


class MyEvaluator:

    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:

        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_args, finetuning_args)
        self.tokenizer.padding_side = "right" # avoid overflow issue in batched inference for llama2
        self.model = dispatch_model(self.model)
        self.template = get_template_and_fix_tokenizer(self.data_args.template, self.tokenizer)

        
        self.eval_template = get_eval_template(self.eval_args.lang)
        #self.choice_inputs = self._encode_choices() ABCD
    '''
    def _encode_choices(self) -> List[int]:#这个函数就是得到choice的token，也就是ABCD的token 就一个
        if isinstance(getattr(self.tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=False)
        return [self.tokenizer.encode(self.eval_template.prefix + ch, **kwargs)[-1] for ch in CHOICES]
    '''

    @torch.inference_mode()
    def batch_inference(self, batch_input: Dict[str, torch.Tensor]) -> List[str]:  #这个函数就是对模型输入一个batch，得到一个batch的输出，输出是每个样本对于每个选项哪个logits最高
        output_ids = self.model.generate(
        input_ids=batch_input['input_ids'],
        attention_mask=batch_input['attention_mask'],
        max_length=2048,  # Adjust max_length as needed
        do_sample=False,
        temperature=0.5  # Set temperature to 0.0 for deterministic output
    ).tolist()
        #print(len(output_ids[0]))
    # Slice off the input part from each output sequence
        real_output_ids = [output_id[len(batch_input['input_ids'][i]):] for i, output_id in enumerate(output_ids)]
        #print(len(real_output_ids[0]))
    # Decode the real output ids to strings
        output_strs = self.tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)

        return output_strs
    '''
        generated_sequences = self.model.generate(**batch_input)

        output_texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_sequences]

        print(output_texts)
        exit()
        logits = self.model(**batch_input).logits

        lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
        choice_probs = torch.nn.functional.softmax(word_probs[:, self.choice_inputs], dim=-1).detach()
        return [chr(ord("A") + offset.item()) for offset in torch.argmax(choice_probs, dim=-1)]
    '''


    def eval(self) -> None: #主函数
        if "token" in inspect.signature(cached_file).parameters:
            kwargs = {"token": self.model_args.hf_hub_token}
        elif "use_auth_token" in inspect.signature(cached_file).parameters: # for transformers==4.31.0
            kwargs = {"use_auth_token": self.model_args.hf_hub_token}
        '''
        mapping = cached_file(  #这里是读取mapping.json来判断一个任务的subject
            path_or_repo_id = os.path.join(self.eval_args.task_dir, self.eval_args.task),
            filename="mapping.json",
            cache_dir=self.model_args.cache_dir,
            **kwargs
        ) 

        with open(mapping, "r", encoding="utf-8") as f:
            categorys: Dict[str, Dict[str, str]] = json.load(f)#这里是读取mapping.json来帮助判断一个任务的category 实际用不到


        category_corrects = {subj: np.array([], dtype="bool") for subj in SUBJECTS}  #读取subjects，应该是要把category映射到subjects

        pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
        
        results = {}
        for subject in pbar: #读取每个数据集的三个子集，self.data_args.split是shell输入的那个
            dataset = load_dataset(
                path=os.path.join(self.eval_args.task_dir, self.eval_args.task),
                name=subject,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.model_args.hf_hub_token
            )
            pbar.set_postfix_str(categorys[subject]["name"])
            inputs, outputs, labels = [], [], []


            for i in trange(len(dataset[self.data_args.split]), desc="Formatting batches", position=1, leave=False):#选中的子集中的每个样本
                support_set = dataset["train"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["train"]))))#选择训练集中随机样本做fewshot
                query, resp, history = self.eval_template.format_example(
                    target_data=dataset[self.data_args.split][i],
                    support_set=support_set,
                    subject_name=categorys[subject]["name"],
                    use_history=self.template.use_history
                )


                input_ids, _ = self.template.encode_oneturn( #这里是使用大模型的模板把prompt加到大模型模板里，再encode 成token ids
                    tokenizer=self.tokenizer, query=query, resp=resp, history=history
                )
                inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                labels.append(resp)

            for i in trange(0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False):
                batch_input = self.tokenizer.pad( #填充
                    inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                ).to(self.model.device)
                preds = self.batch_inference(batch_input)#推理
                outputs += preds

            corrects = (np.array(outputs) == np.array(labels))
            category_name = categorys[subject]["category"]
            category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
            results[subject] = {str(i): outputs[i] for i in range(len(outputs))}

        pbar.close()
        self._save_results(category_corrects, results)
        
        dataset = load_dataset(
            path=,
            cache_dir=self.model_args.cache_dir,
            download_mode=self.eval_args.download_mode,
            token=self.model_args.hf_hub_token
        )
        '''
        # 


        # 设置 JSON 文件的路径
        train_json_path = 'data/json_output/train.json'
        test_json_path = 'data/json_output/test.json'


        # 加载每个数据集分割
        train_dataset = load_dataset('json', data_files=train_json_path)
        test_dataset = load_dataset('json', data_files=test_json_path)

        train_dataset=load_dataset('csv',data_files='data/combined_output_100.csv')
        test_dataset=load_dataset('csv',data_files='data/combined_output_100.csv')
        # 创建 DatasetDict
        dataset= DatasetDict({
            'train': train_dataset['train'],
            'test': test_dataset['train']
        })

        inputs, outputs, labels = [], [], []


        for i in trange(len(dataset[self.data_args.split]), desc="Formatting batches", position=1, leave=False):#选中的子集中的每个样本
            support_set = dataset["train"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["train"]))))#选择训练集中随机样本做fewshot
            query, resp, history = self.eval_template.format_example(
                target_data=dataset[self.data_args.split][i],
                support_set=support_set,
                #subject_name=categorys[subject]["name"],
                use_history=self.template.use_history
            )

            #print(query,"/n",resp,'/n',history)
            #exit()

            input_ids, _ = self.template.encode_oneturn( #这里是使用大模型的模板把prompt加到大模型模板里，再encode 成token ids
                tokenizer=self.tokenizer, query=query, resp=resp, history=history
            )
            inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
            labels.append(resp)

        for i in trange(0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False):
            batch_input = self.tokenizer.pad( #填充
                inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
            ).to(self.model.device)
            preds = self.batch_inference(batch_input)#推理
            '''
            test=self.normalize_answer(preds)
            print("##################")
            print(labels[0])
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(test)
            print("###########")
            exit()
            '''
            outputs.extend(preds)


        
        normalized_outputs = [self.normalize_answer(output) for output in outputs]
        normalized_labels = [self.normalize_answer(label) for label in labels]
        
        matches = []
        meaningless_resp=0
        for output in normalized_outputs:
            output_words = set(output.split())
            match_found = False
            for word in output_words:
                if word in normalized_labels:
                    matches.append(word)  # Append the matching label
                    match_found = True
                    break  # Break the loop once a match is found
            if not match_found:
                matches.append('neutral')# Append 'Neutral' if no label matches
                meaningless_resp+=1
        
                  



        performance= self.calculate_multiclass_metrics(matches,normalized_labels)
        print(performance)
        print("wrong output is:",meaningless_resp)
        self.save_to_json(performance,normalized_outputs,normalized_labels)

    
    def calculate_multiclass_metrics(self,matches, labels):
        # Initialize and fit a label encoder to the unique labels
        encoder = LabelEncoder()
        unique_labels = set(matches + labels)
        encoder.fit(list(unique_labels))

        # Encode matches and labels
        encoded_matches = encoder.transform(matches)
        encoded_labels = encoder.transform(labels)

        # Calculate accuracy
        accuracy = accuracy_score(encoded_labels, encoded_matches)

        # Calculate micro-average metrics
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(encoded_labels, encoded_matches, average='micro')

        # Calculate macro-average and weighted-average metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(encoded_labels, encoded_matches, average='macro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(encoded_labels, encoded_matches, average='weighted')

        # Calculate metrics for each class
        precision, recall, f1, _ = precision_recall_fscore_support(encoded_labels, encoded_matches, average=None, labels=range(len(unique_labels)))
        class_metrics = {label: {'precision': p, 'recall': r, 'f1_score': f} for label, p, r, f in zip(encoder.classes_, precision, recall, f1)}

        # Prepare and return metrics
        overall_metrics = {
            'accuracy': accuracy,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_score_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_score_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_score_weighted': f1_weighted
        }

        return {'class_metrics': class_metrics, 'overall_metrics': overall_metrics}
            

        
    '''
        corrects = (np.array(outputs) == np.array(labels))
        category_name = categorys[subject]["category"]
        category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
        category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
        results[subject] = {str(i): outputs[i] for i in range(len(outputs))}
    '''


    # adapted the flowing from Squad v1.1 evaluation, without removing the articles.
    def normalize_answer(self,s):
        """Lower text and remove punctuation, and extra whitespace."""

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))


    def save_to_json(self, dict_data, list1, list2, dict_filename='performance.json', list_filename='outputs.json'):
        # Convert model to string for the path
        model_str = str(self.model_args.model_name_or_path)
        prompt_str=str(self.eval_args.lang)
        temperature_str=str(0.1)
        # Construct the directory path
        directory_path = os.path.join(model_str, prompt_str, temperature_str)

        # Create the directory if it doesn't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Save the dictionary to its file
        dict_file_path = os.path.join(directory_path, dict_filename)
        with open(dict_file_path, 'w', encoding='utf-8') as file:
            json.dump(dict_data, file, ensure_ascii=False, indent=4)

        # Save the lists to another file
        lists_file_path = os.path.join(directory_path, list_filename)
        with open(lists_file_path, 'w', encoding='utf-8') as file:
            json.dump({"list1": list1, "list2": list2}, file, ensure_ascii=False, indent=4)

        print(f"Dictionary data saved to {dict_file_path}")
        print(f"List data saved to {lists_file_path}")


'''
    def _save_results(self, category_corrects: Dict[str, np.ndarray], results: Dict[str, Dict[int, str]]) -> None:
        score_info = "\n".join([
            "{:>15}: {:.2f}".format(category_name, 100 * np.mean(category_correct))
            for category_name, category_correct in category_corrects.items() if len(category_correct)
        ])
        print("score_info"+"#"*50)
        print(score_info)

        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=False)
            with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f, indent=2)

            with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)
'''  

if __name__ == "__main__":
    evaluator = MyEvaluator()
    evaluator.eval()

# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py

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


class Evaluator:

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
        max_length=512,  # Adjust max_length as needed
        do_sample=False,
        temperature=0  # Set temperature to 0.0 for deterministic output
    ).tolist()
        print(len(output_ids[0]))
    # Slice off the input part from each output sequence
        real_output_ids = [output_id[len(batch_input['input_ids'][i]):] for i, output_id in enumerate(output_ids)]
        print(len(real_output_ids[0]))

    # Decode the real output ids to strings
        output_strs = self.tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)
        print(output_strs[0])
        return (output_strs[0])

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
        file_path = '/aul/homes/hzhen011/LLaMA-Factory/evaluation/combined_output.csv'
        dataset = load_dataset('csv', data_files=file_path)

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


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.eval()

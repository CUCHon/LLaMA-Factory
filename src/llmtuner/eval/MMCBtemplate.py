from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

from llmtuner.extras.constants import CHOICES

if TYPE_CHECKING:
    from datasets import Dataset


@dataclass
class EvalTemplate:

    system: str
  #  choice: str
    answer: str
    prefix: str


    '''
    def parse_example(
        self,
        example: Dict[str, str] #输入一个字典，包含question 和 A B C d answer
    ) -> Tuple[str, str]:
        candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in CHOICES if ch in example]#这里是保存每个选项还有每个选项对应的答案（有换行） 我们的数据集没必要留ABCD了

        return "".join([example["question"]] + candidates + [self.answer]), example["answer"]# 前面是返回 问题/n选项 答案/n 一个"Answer："。后面是返回答案对应的选项
    '''
    def parse_example(
        self,
        example: Dict[str, str] #输入一个字典，包含question 和 A B C d answer
    ) -> Tuple[str, str]:
        query, resp= example['Text'], example['Sentiment']
        return query+self.answer, resp
        
    def parse_example(
        self,
        example: Dict[str, str] #输入一个字典，包含question 和 A B C d answer
    ) -> Tuple[str, str]:
        query, resp= example['Text'], example['Sentiment']
        return query+self.answer, resp


    def format_example(  #根据注册的模板来格式化样例
        self,
        target_data: Dict[str, str],
        support_set: "Dataset",
        #subject_name: str,
        use_history: bool
    ) -> Tuple[str, str, List[Tuple[str, str]]]:

        #query, resp= target_data['Text'], target_data['Neutral/Change/Sustain']


        query, resp = self.parse_example(target_data)# 前面是返回 text加一个"Answer："。后面是返回标签
        history = [self.parse_example(support_set[k]) for k in range(len(support_set))] #exist only n_shot>0 list of examples

        if len(history):
            temp = history.pop(0)

            history.insert(0, (self.system + temp[0], temp[1])) #这里是fewshot第一个样本加上模板里的system prompt


        else:
            query = self.system + query   #这里是zeroshot  加上模板里的system prompt

        if not use_history: #默认false 默认启动 我猜这里是默认single turn
            query = "\n\n".join(["".join(item) for item in history] + [query])
            history = []
        return query.strip(), resp, history


eval_templates: Dict[str, EvalTemplate] = {}


def register_eval_template(
    name: str,
    system: str,
    #choice: str,
    answer: str,
    prefix: str
) -> None:
    eval_templates[name] = EvalTemplate(
        system=system,
        #choice=choice,
        answer=answer,
        prefix=prefix
    )


def get_eval_template(name: str) -> EvalTemplate:
    eval_template = eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

'''
register_eval_template(
    name="en",
    system="The following are multiple choice questions (with answers) about {subject}.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" "
)

'''


register_eval_template(
    name="psy",
    system="Given a Motivational Interview record between an alcohol-abusing patient and a psychotherapist, please determine the patient's attitude about reducing the alcohol use. \n From:\n1.  Neutral \n2. Positive(If the patient has a tendency to reduce alcohol consumption )\n3. Negative (If the patient tend to maintain alcohol consumption)\n ",
    #choice="\n{choice}. {content}",
    answer="\nAnswer: The attitude is：",
    prefix="\n"
)


register_eval_template(
    name="1cot",
    system="Given a Motivational Interview record between an alcohol-abusing patient and a psychotherapist, please determine the patient's attitude about reducing the alcohol use. \n From:\n1.  Neutral \n2. Positive(If the patient has a tendency to reduce alcohol consumption )\n3. Negative (If the patient tend to maintain alcohol consumption)\n ",
    #choice="\n{choice}. {content}",
    answer="\nAnswer: The attitude is：",
    prefix="\n"
)

register_eval_template(
    name="zcot",
    system="Given a Motivational Interview record between an alcohol-abusing patient and a psychotherapist, please determine the patient's attitude about reducing the alcohol use. \n From:\n1.  Neutral \n2. Positive(If the patient has a tendency to reduce alcohol consumption )\n3. Negative (If the patient tend to maintain alcohol consumption)\n ",
    #choice="\n{choice}. {content}",
    answer="\nAnswer: Let's think step by step, ",
    prefix="\n"
)

register_eval_template(
    name="ztm",
    system="Given a Motivational Interview record between an alcohol-abusing patient and a psychotherapist, please determine the patient's attitude about reducing the alcohol use. \n From:\n1.  Neutral \n2. Positive(If the patient has a tendency to reduce alcohol consumption )\n3. Negative (If the patient tend to maintain alcohol consumption)\n ",
    #choice="\n{choice}. {content}",
    answer="\nAnswer: The attitude is：",
    prefix="\n"
)

register_eval_template(
    name="fr",
    system="Sur la base des notes médicales fournies, diagnostiquer le patient-----Death/survival--\n\n",
   # choice="\n{choice}. {content}",
    answer="\nrépondre：",
    prefix="\n"
)

register_eval_template(
    name="pt",
    system="Com base nas notas médicas fornecidas, diagnosticar o paciente----Death/survival---\n\n",
  #  choice="\n{choice}. {content}",
    answer="\nresposta：",
    prefix="\n"
)


register_eval_template(
    name="zh",
    system="基于这个病人的医疗记录 请诊断他的病症------\n\n",
   # choice="\n{choice}. {content}",
    answer="\n诊断：",
    prefix="\n"
)


register_eval_template(
    name="en",
    system="Given the medical notes, please diagnose the patient---Death/survival----\n\n",
    #choice="\n{choice}. {content}",
    answer="\nanswer：",
    prefix="\n"
)

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

from llmtuner.extras.constants import CHOICES

if TYPE_CHECKING:
    from datasets import Dataset


@dataclass
class EvalTemplate:

    system: str
    choice: str
    answer: str
    prefix: str

    def parse_example(
        self,
        example: Dict[str, str] #输入一个字典，包含question 和 A B C d answer
    ) -> Tuple[str, str]:
        candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in CHOICES if ch in example]#这里是保存每个选项还有每个选项对应的答案（有换行） 我们的数据集没必要留ABCD了

        return "".join([example["question"]] + candidates + [self.answer]), example["answer"]# 前面是返回 问题/n选项 答案/n 一个"Answer："。后面是返回答案对应的选项

    def format_example(
        self,
        target_data: Dict[str, str],
        support_set: "Dataset",
        subject_name: str,
        use_history: bool
    ) -> Tuple[str, str, List[Tuple[str, str]]]:



        query, resp = self.parse_example(target_data)# 前面是返回 问题/n选项 答案/n 一个"Answer："。后面是返回答案对应的选项比如A


        history = [self.parse_example(support_set[k]) for k in range(len(support_set))] #exist only n_shot>0 list of examples

        if len(history):
            temp = history.pop(0)


            
            history.insert(0, (self.system.format(subject=subject_name) + temp[0], temp[1])) #这里是加上模板里的system prompt

        else:
            query = self.system.format(subject=subject_name) + query   #这里是zeroshot  加上模板里的system prompt

        if not use_history: #默认false 默认启动 我猜这里是默认single turn
            query = "\n\n".join(["".join(item) for item in history] + [query])
            history = []
        return query.strip(), resp, history


eval_templates: Dict[str, EvalTemplate] = {}


def register_eval_template(
    name: str,
    system: str,
    choice: str,
    answer: str,
    prefix: str
) -> None:
    eval_templates[name] = EvalTemplate(
        system=system,
        choice=choice,
        answer=answer,
        prefix=prefix
    )


def get_eval_template(name: str) -> EvalTemplate:
    eval_template = eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template


register_eval_template(
    name="en",
    system="The following are multiple choice questions (with answers) about {subject}.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" "
)


register_eval_template(
    name="zh",
    system="以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    prefix="\n"
)

register_eval_template(
    name="mimic",
    system="Given the following facts from a medical。Please ----------\n\n",
    choice="\n{choice}. {content}",
    answer="\nanswer：",
    prefix="\n"
)


register_eval_template(
    name="french",
    system="Compte tenu des faits suivants, tirés d'un rapport médical。Please----------\n\n",
    choice="\n{choice}. {content}",
    answer="\nrépondre：",
    prefix="\n"
)
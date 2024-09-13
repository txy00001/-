import dataclasses
from typing import List


@dataclasses.dataclass
class Conversation:
    begin: str = "<s>"
    end: str = "</s>"
    user_begin: str = "[INST]"
    user_end: str = "[/INST]"
    sys_begin: str = "<<SYS>>"
    sys_end: str = "<</SYS>>"
    roles: List[str] = dataclasses.field(default_factory=lambda: ["USER", "ASSISTANT"])
    sys: str = " "
    messages: List[List[str]] = dataclasses.field(default_factory=list)

    def append_message(self, role: str, message: str):
        self.messages.append([role, message])

    def get_prompt(self):
        dialog = []
        for i, ((user, question), (assist, answer)) in enumerate(
            zip(self.messages, self.messages[1:])
        ):
            if i == 0:
                assert user == self.roles[0]
                dialog.append(
                    f"{self.begin}{self.user_begin} {self.sys_begin} {self.sys} {self.sys_end} {question} {self.user_end} {answer}{self.end}"
                )
            else:
                dialog.append(
                    f"{self.begin}{self.user_begin} {question} {self.user_end} {answer}{self.end}"
                )
        return "\n".join(dialog)

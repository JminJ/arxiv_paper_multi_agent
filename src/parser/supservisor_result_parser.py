from langchain_core.messages import AIMessage

def parsing_supervisor_result(result_message:AIMessage):
    """supervisor agent의 결과 메세지 내에서 next role을 추출해 반환하는 parser

    Args:
        result_message (AIMessage): supervisor agent의 결과 메세지
    """
    next_role = result_message.additional_kwargs["function_call"]["arguments"]
    next_role = eval(next_role)["next"]

    next_message = [{"output": next_role}]

    return next_message
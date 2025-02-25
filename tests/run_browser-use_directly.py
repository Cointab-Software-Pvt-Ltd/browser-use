import asyncio
import json
import os
import re

import pyautogui

from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser, BrowserConfig, BrowserContextConfig
from browser_use.browser.context import (
    BrowserContextWindowSize,
)
from browser_use.support import utils

_global_browser = None
_global_browser_context = None
_global_agent = None

_global_history_file_path = "history.json"
_global_secret_file_path = "secret.json"

window_width = pyautogui.size()[0] * 0.8
window_height = pyautogui.size()[1] * 0.8


def resolve_sensitive_env_variables(text):
    if not text:
        return text
    env_vars = re.findall(r'\$SENSITIVE_[A-Za-z0-9_]*', text)
    result = text
    for var in env_vars:
        env_name = var[1:]
        env_value = os.getenv(env_name)
        if env_value is not None:
            result = result.replace(var, env_value)
    return result


async def run_org_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        history_file_input,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method
):
    final_result = None
    errors = None
    model_actions = None
    model_thoughts = None
    history_file = None
    trace_file = None
    try:
        global _global_browser, _global_browser_context, _global_agent, _global_secret_file_path
        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        if use_own_browser:
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        if _global_browser is None:
            _global_browser = Browser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )
        if task is None:
            task = ""

        if _global_agent is None:
            sensitive_data = None
            if os.path.exists(_global_secret_file_path):
                sensitive_data = json.load(open(_global_secret_file_path, 'r'))
            _global_agent = Agent(
                task=task,
                sensitive_data=sensitive_data,
                llm=llm,
                use_vision=use_vision,
                browser=_global_browser,
                browser_context=_global_browser_context,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        if history_file_input is None:
            history = await _global_agent.run(max_steps=max_steps)

            history_file = os.path.join(save_agent_history_path, f"{_global_agent.state.agent_id}.json")
            _global_agent.save_history(history_file)

            final_result = history.final_result()
            errors = history.errors()
            model_actions = history.model_actions()
            model_thoughts = history.model_thoughts()
            trace_file = utils.get_latest_files(save_trace_path).get('.zip')
        else:
            await _global_agent.load_and_rerun(history_file_input)

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
    finally:
        _global_agent = None
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None
            if _global_browser:
                await _global_browser.close()
                _global_browser = None
        latest_video = None
        if save_recording_path is not None:
            latest_video = utils.get_latest_files(save_recording_path).get('.webm')
        return final_result, errors, model_actions, model_thoughts, trace_file, history_file, latest_video


async def run_browser_agent(
        task=None,
        history_file_input=None,
        llm_provider="openai",
        llm_model_name="gpt-4o",
        llm_num_ctx=32000,
        llm_temperature=0.7,
        llm_base_url='',
        llm_api_key='',
        use_own_browser=False,
        keep_browser_open=True,
        headless=False,
        disable_security=False,
        window_w=window_width,
        window_h=window_height,
        save_recording_path="./tmp/record_videos",
        save_agent_history_path="./tmp/agent_history",
        save_trace_path="./tmp/traces",
        enable_recording=True,
        max_steps=1000,
        use_vision=True,
        max_actions_per_step=50,
        tool_calling_method="auto"
):
    global _global_history_file_path
    try:
        if not enable_recording:
            save_recording_path = None

        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)

        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )

        if history_file_input is None:
            task = resolve_sensitive_env_variables(task)

        (final_result, errors, model_actions, model_thoughts,
         trace_file, history_file, latest_video) = await run_org_agent(
            llm=llm,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            task=task,
            history_file_input=history_file_input,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method
        )

        print(latest_video)
        if history_file_input is None and history_file is not None:
            print(history_file)
            data = json.load(open(history_file, 'r'))["history"]
            data = utils.remove_error_from_history(data)
            current = []
            if os.path.exists(_global_history_file_path):
                current = json.load(open(_global_history_file_path, 'r'))["history"]
                current = utils.remove_completed_from_history(current)
            final = current + data
            utils.fix_history_step_numbers(final)
            utils.move_xpath_from_interacted_element_to_action(final)
            final = {"history": final}
            json.dump(final, open(_global_history_file_path, 'w'))
            print(model_actions)

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        print(errors)


async def exec_tasks():
    is_new = True
    if os.path.exists(_global_history_file_path):
        current = json.load(open(_global_history_file_path, 'r'))["history"]
        current = utils.remove_error_from_history(current)
        utils.fix_history_step_numbers(current)
        utils.move_xpath_from_interacted_element_to_action(current)
        current = {"history": current}
        json.dump(current, open(_global_history_file_path, 'w'))
        await run_browser_agent(history_file_input=_global_history_file_path)
        print("Steps Done, and saved for replay")
        temp = utils.remove_completed_from_history(current["history"])
        for idx, step in enumerate(temp):
            is_new = False
            print(str(idx + 1), step["model_output"]["current_state"]["next_goal"])
    while True:
        if is_new:
            print("Enter new task: ")
        else:
            print("Enter additional task: ")
        is_new = False
        task_text = ""
        while True:
            partial_task_input = input()
            if partial_task_input != "-":
                task_text = task_text + "\n" + partial_task_input
                task_text = task_text.strip()
            else:
                break
        if len(task_text) > 0:
            await run_browser_agent(task=task_text)
            print("Steps Done, and saved for replay")
            current = json.load(open(_global_history_file_path, 'r'))["history"]
            current = utils.remove_completed_from_history(current)
            for idx, step in enumerate(current):
                print(str(idx + 1), step["model_output"]["current_state"]["next_goal"])


asyncio.run(exec_tasks())

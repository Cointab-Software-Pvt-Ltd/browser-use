from typing import Optional

from pydantic import BaseModel, model_validator


# Action Input Models


class GoToUrlAction(BaseModel):
    url: str


class RequestAction(BaseModel):
    action_name: str
    action_description: str


class IfConditionAction(BaseModel):
    condition_lhs: str
    operator: str
    condition_rhs: str


class ClickElementAction(BaseModel):
    index: int
    xpath: Optional[str] = None
    right_click: Optional[bool] = False


class PhysicalClickElementAction(BaseModel):
    x: int
    y: int
    right_click: Optional[bool] = False
    long_press: Optional[bool] = False
    click_count: Optional[int] = 1
    press_duration: Optional[int] = 1


class InputTextAction(BaseModel):
    index: int
    text: Optional[str] = None
    secret_key: Optional[str] = None
    xpath: Optional[str] = None
    has_human_keystroke: Optional[bool] = True


class DoneAction(BaseModel):
    text: str
    success: bool


class SwitchTabAction(BaseModel):
    page_id: int


class OpenTabAction(BaseModel):
    url: str


class ScrollAction(BaseModel):
    amount: Optional[int] = None  # The number of pixels to scroll. If None, scroll down/up one page


class GenerateTOTP(BaseModel):
    totp_key: str
    save_in_secret_with_key: Optional[str] = None


class SendKeysAction(BaseModel):
    keys: str


class ExtractPageContentAction(BaseModel):
    value: str


class NoParamsAction(BaseModel):
    """
    Accepts absolutely anything in the incoming data
    and discards it, so the final parsed model is empty.
    """

    @model_validator(mode='before')
    def ignore_all_inputs(cls, values):
        # No matter what the user sends, discard it and return empty.
        return {}

    class Config:
        # If you want to silently allow unknown fields at top-level,
        # set extra = 'allow' as well:
        extra = 'allow'

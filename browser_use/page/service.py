from rebrowser_playwright.async_api import (
    Page,
)


class BrowserPage:
    def __init__(self, page: Page, context, session, page_no: int):
        self.page = page
        self.context = context
        self.session = session
        self.page_no = page_no
        self.stream_client = None

    async def stream_screenshot(self, params):
        params["tab_no"] = self.page_no
        if self.context.config.screenshot_capture is not None:
            await self.context.config.screenshot_capture(params)
        await self.stream_client.send('Page.screencastFrameAck', {
            "sessionId": params["sessionId"],
        })

    async def create_stream(self):
        if self.context.config.screenshot_capture is not None:
            self.stream_client = await self.session.context.new_cdp_session(self.page)
            await self.stream_client.send("Page.startScreencast", {"format": "jpeg"})
            self.stream_client.on("Page.screencastFrame", lambda params: self.stream_screenshot(params))

    async def close(self):
        await self.page.close()

    def decrease_page_no(self):
        self.page_no -= 1

    def get_page_no(self):
        return self.page_no

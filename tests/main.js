let img = document.getElementById("video");
let container = document.getElementById("buttonContainer");
let browser_ts = document.getElementById("browser_ts");
let buttonCount = 0;
let buttonSelected = 0;

ws = new WebSocket("ws://localhost:8089");
ws.onmessage = (event) => {
    let payload = JSON.parse(event.data);
    if (!payload.fn || payload.fn == 'frame') {
        while (payload.page_count != buttonCount) {
            if (payload.page_count > buttonCount) {
                const btn = document.createElement("button");
                btn.textContent = `Button ${payload.tab_no}`;
                btn.id = `btn-${payload.tab_no}`;
                btn.onclick = () => buttonSelected = btn.id.split('-')[1];
                container.appendChild(btn);
                buttonCount++;
            } else {
                if (container.lastChild) {
                    container.removeChild(container.lastChild);
                    buttonCount--;
                }
                if (payload.tab_no < buttonSelected)
                    buttonSelected = payload.tab_no;
            }
        }
        if (payload.tab_no == buttonSelected) {
            img.setAttribute("src", `data:image/jpeg;base64,${payload.frame}`);
        }
        browser_ts.textContent = new Date(payload.ts).toString();
    } else if (payload.fn == 'size') {
        img.setAttribute("height", payload.size.height + "px");
        img.setAttribute("width", payload.size.width + "px");
    }
};


function updateCounters(selectedText) {
    let wordcount = 0;
    let spacecount = 0;
    let sentcount = 0;

    if (selectedText) {
        // Trim trailing spaces
        selectedText = selectedText.trim();

        for (let i = 0; i < selectedText.length; i++) {
            if (selectedText[i] === ' ') {
                if (selectedText[i + 1] !== ' ') {
                    wordcount++;
                }
                spacecount++;
            }

            if (selectedText[i] === '.') {
                sentcount++;
            }
        }
    }

    // Send the counters back to the popup
    chrome.runtime.sendMessage({
        action: "updatePopup",
        counters: {
            space: spacecount,
            chars: selectedText ? selectedText.length - spacecount - sentcount : 0,
            words: selectedText ? (selectedText.length === 0 ? 0 : wordcount + 1) : 0,
            sentences: sentcount
        }
    });
}

// Listen for messages from the popup
chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    if (request.action === "getSelectedText") {
        const selectedText = window.getSelection().toString();
        updateCounters(selectedText);
    }
});



// Execute the content script and get the selected text
chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    chrome.tabs.executeScript(tabs[0].id, { file: "wordcount.js" }, function () {
        // Send a message to the content script to get the selected text
        chrome.tabs.sendMessage(tabs[0].id, { action: "getSelectedText" });
    });
});

// Listen for the response from the content script
chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    if (request.action === "updatePopup") {
        const counters = request.counters;
        updatePopup(counters);
    }
});

function updatePopup(counters) {
    const spaceElem = document.getElementById('space');
    const charsElem = document.getElementById('chars');
    const wordsElem = document.getElementById('words');
    const sentencesElem = document.getElementById('sentences');

    if (counters) {
        spaceElem.textContent = counters.space;
        charsElem.textContent = counters.chars;
        wordsElem.textContent = counters.words;
        sentencesElem.textContent = counters.sentences;
    } else {
        // Handle the case where no text is selected
        spaceElem.textContent = 0;
        charsElem.textContent = 0;
        wordsElem.textContent = 0;
        sentencesElem.textContent = 0;
    }
}

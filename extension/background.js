// Create a context menu item
chrome.contextMenus.create({
    id: "fakeNewsDetector",
    title: "Fake News Detector",
    contexts: ["selection"],  // Show the menu item when text is selected
});
// Add a listener for the context menu item
chrome.contextMenus.onClicked.addListener(function(info, tab) {
    if (info.menuItemId === "fakeNewsDetector") {
        // Send a message to the content script to get the selected text
        chrome.tabs.sendMessage(tab.id, { action: "getSelectedText" });
    }
});
// Make external links open in new tab
document.addEventListener('DOMContentLoaded', function() {
    var links = document.links;
    
    for (var i = 0, linksLength = links.length; i < linksLength; i++) {
        if (links[i].hostname != window.location.hostname) {
            links[i].target = '_blank';
            links[i].rel = 'noopener noreferrer';
        }
    }
});

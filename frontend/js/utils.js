const Utils = {
    show(id) {
        document.getElementById(id)?.classList.remove('hidden');
    },
    
    hide(id) {
        document.getElementById(id)?.classList.add('hidden');
    },
    
    scrollTo(id) {
        document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    },
    
    getRiskIcon(level) {
        return { 'Low': '✅', 'Medium': '⚠️', 'High': '🚨' }[level] || '❓';
    }
};
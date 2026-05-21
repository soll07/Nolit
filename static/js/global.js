// CSRF 쿠키 헬퍼 (fetch API에서 공통 사용)
function getCookie(name) {
    const v = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
    return v ? v.pop() : '';
}

// 로그아웃 시 로컬스토리지 비움
window.clearPersonaLocalStorage = function() {
    console.log("logout clicked");
    localStorage.removeItem("nolit-groups");
}
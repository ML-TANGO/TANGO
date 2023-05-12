export const userIDRule = () => {
  // const regex = new RegExp("/^[A-Za-z0-9+]{5,20}$/");
  const userIdRegex = /^[A-Za-z0-9+]{5,20}$/;

  return [v => userIdRegex.test(v) || `5~20자의 영문 소문자, 숫자와 특수기호(_),(-)만 사용 가능합니다.`];
};

export const emailRule = () => {
  // const regex = new RegExp("/^[A-Za-z0-9+]{5,20}$/");
  const userIdRegex = /^([\w-]+(?:\.[\w-]+)*)@((?:[\w-]+\.)*\w[\w-]{0,66})\.([a-z]{2,6}(?:\.[a-z]{2})?)$/i;

  return [v => userIdRegex.test(v) || `이메일 형식이 잘못되었습니다.`];
};

export const passwordRule = () => {
  // const regex = new RegExp("/^[A-Za-z0-9+]{5,20}$/");
  const userIdRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*[$@$!%*?&])[A-Za-z\d$@$!%*?&]{8,}/;

  return [v => userIdRegex.test(v) || `8자 이상, 영문 대소문자, 특수문자를 사용하세요.`];
};

export const passwordConfirmRule = password => {
  return [v => password === v || `비밀번호가 일치하지 않습니다.`];
};

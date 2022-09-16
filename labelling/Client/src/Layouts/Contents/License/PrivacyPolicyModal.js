import React from "react"
import { Container, Row, Col, Modal, ModalHeader, ModalBody } from "reactstrap"

function PrivacyPolicyModal(props) {
  const { modal, toggle } = props
  return (
    <Modal
      isOpen={modal}
      toggle={toggle}
      modalClassName={"ltr-support"}
      contentClassName="pt-4"
      classNameName={"modal-dialog--primary modal-dialog--header"}
      size={"xl"}
    >
      <ModalHeader toggle={toggle}>개인정보 처리 방침</ModalHeader>
      <ModalBody>
        <Container>
          <Row>
            <Col md={12} xl={12}>
              <p className="text-left">
                <p className="ls2 lh6 bs5 ts4">
                  <em className="emphasis">{`<(주)웨다>('AI 비전 품질관리 솔루션'이하  'BluAi')`}</em>은(는) 개인정보보호법에 따라 이용자의
                  개인정보 보호 및 권익을 보호하고 개인정보와 관련한 이용자의 고충을 원활하게 처리할 수 있도록 다음과 같은 처리방침을 두고
                  있습니다.
                </p>
                <p className="ls2 lh6 bs5 ts4">
                  <em className="emphasis">{`<(주)웨다>('BluAi')`}</em> 은(는) 회사는 개인정보처리방침을 개정하는 경우 웹사이트
                  공지사항(또는 개별공지)을 통하여 공지할 것입니다.
                </p>
                <p className="ls2">
                  ○ 본 방침은부터 <em className="emphasis">2020</em>년 <em className="emphasis">12</em>월 <em className="emphasis">28</em>
                  일부터 시행됩니다.
                </p>
                <br />
                <p className="lh6 bs4">
                  <strong>
                    1. 개인정보의 처리 목적 <em className="emphasis">{`<(주)웨다>('AI 비전 품질관리 솔루션' 이하  'BluAi')`}</em>은(는)
                    개인정보를 다음의 목적을 위해 처리합니다. 처리한 개인정보는 다음의 목적이외의 용도로는 사용되지 않으며 이용 목적이
                    변경될 시에는 사전동의를 구할 예정입니다.
                  </strong>
                </p>
                <ul className="list_indent2 mgt10">
                  <p className="ls2">가. 라이센스 발급</p>
                  <p className="ls2">라이센스 발급 등을 목적으로 발급에 필요한 개인정보를 처리합니다.</p>
                  <p className="ls2">나. 마케팅 및 광고에의 활용</p>
                  <p className="ls2">신규 서비스(제품) 개발 및 맞춤 서비스 제공 등을 목적으로 개인정보를 처리합니다.</p>
                </ul>
                {/* <br /> */}
                {/* <p className="sub_p mgt30">
                  <strong>2. 개인정보 파일 현황</strong>
                </p>
                <ul className="list_indent2 mgt10">
                  <li className="tt">
                    <b>1. 개인정보 파일명 : BluAi 라이센스 정보</b>
                  </li>
                  <li>개인정보 항목 : 이메일, 이름, 직책, 부서, 회사명</li>
                  <li>수집방법 : 라이센스 신청 입력 란에 본인이 직접 입력</li>
                  <li>보유근거 : 라이센스 발급을 위한 필수 정보 입력 및 마케팅 정보 활용</li>
                  <li>보유기간 : 3년</li>
                </ul> */}
                <br />
                <p className="lh6 bs4">
                  <strong>2. 개인정보의 처리 및 보유 기간</strong>
                  <br />
                  <br />① <em className="emphasis">{`<(주)웨다>('BluAi')`}</em>은(는) 법령에 따른 개인정보 보유·이용기간 또는 정보주체로부터
                  개인정보를 수집시에 동의 받은 개인정보 보유,이용기간 내에서 개인정보를 처리,보유합니다.
                  <br />
                  <br />② 각각의 개인정보 처리 및 보유 기간은 다음과 같습니다.
                </p>
                <ul className="list_indent2 mgt10">
                  <li className="tt">{`가. 라이센스 발급`}</li>
                  <li className="tt">{`<라이센스 발급>과 관련한 개인정보는 수집.이용에 관한 동의일로부터<3년>까지 위 이용목적을 위하여 보유.이용됩니다.`}</li>
                  <li>보유근거 : 라이센스 발급을 위한 필수 정보</li>
                </ul>
                <ul className="list_indent2 mgt10">
                  <li className="tt">{`나. <마케팅 및 광고에의 활용>`}</li>
                  <li className="tt">{`<마케팅 및 광고에의 활용>와 관련한 개인정보는 수집.이용에 관한 동의일로부터<3년>까지 위 이용목적을 위하여 보유.이용됩니다.`}</li>
                  <li>보유근거 : 마케팅 정보 활용</li>
                </ul>
                <br />
                <p className="lh6 bs4">
                  <strong>
                    3. 정보주체와 법정대리인의 권리·의무 및 그 행사방법 이용자는 개인정보주체로써 다음과 같은 권리를 행사할 수 있습니다.
                  </strong>
                </p>
                <p className="ls2">
                  ① 정보주체는 (주)웨다에 대해 언제든지 라이센스 발급을 제외한 마케팅 및 광고에의 활용에 관련하여 개인정보
                  열람,정정,삭제,처리정지 요구 등의 권리를 행사할 수 있습니다.
                </p>
                <p className="sub_p">
                  ② 제1항에 따른 권리 행사는(주)웨다에 대해 개인정보 보호법 시행령 제41조제1항에 따라 서면, 전자우편, 모사전송(FAX) 등을
                  통하여 하실 수 있으며 (주)웨다은(는) 이에 대해 지체 없이 조치하겠습니다.
                </p>
                <p className="sub_p">
                  ③ 제1항에 따른 권리 행사는 정보주체의 법정대리인이나 위임을 받은 자 등 대리인을 통하여 하실 수 있습니다. 이 경우 개인정보
                  보호법 시행규칙 별지 제11호 서식에 따른 위임장을 제출하셔야 합니다.
                </p>
                <p className="sub_p">
                  ④ 개인정보 열람 및 처리정지 요구는 개인정보보호법 제35조 제5항, 제37조 제2항에 의하여 정보주체의 권리가 제한 될 수
                  있습니다.
                </p>
                <p className="sub_p">
                  ⑤ 개인정보의 정정 및 삭제 요구는 다른 법령에서 그 개인정보가 수집 대상으로 명시되어 있는 경우에는 그 삭제를 요구할 수
                  없습니다.
                </p>
                <p className="sub_p">
                  ⑥ (주)웨다은(는) 정보주체 권리에 따른 열람의 요구, 정정·삭제의 요구, 처리정지의 요구 시 열람 등 요구를 한 자가 본인이거나
                  정당한 대리인인지를 확인합니다.
                </p>
                <br />
                <p className="lh6 bs4">
                  <strong>4. 처리하는 개인정보의 항목 작성 </strong>
                  <br />
                  <br /> ① <em className="emphasis">{`<(주)웨다>('AI 비전 품질관리 솔루션' 이하 'BluAi')`}</em>은(는) 다음의 개인정보 항목을
                  처리하고 있습니다.
                </p>
                <ul className="list_indent2 mgt10">
                  <li className="tt">{`가. <라이센스 발급>`}</li>
                  <li>필수항목 : 이메일, 이름, 직책, 부서, 회사명, 하드웨어 고유번호</li>
                </ul>
                <ul className="list_indent2 mgt10">
                  <li className="tt">{`나. <마케팅 및 광고에의 활용>`}</li>
                  <li>필수항목 : 이메일, 이름, 직책, 부서, 회사명</li>
                </ul>
                <br />
                <p className="lh6 bs4">
                  <strong>
                    5. 개인정보의 파기<em className="emphasis">{`<(주)웨다>('BluAi')`}</em>
                    {`은(는) 원칙적으로 개인정보 처리목적이 달성된 경우에는 지체없이 해당 개인정보를 파기합니다. 파기의 절차, 기한 및 방법은 다음과 같습니다.`}
                  </strong>
                </p>
                <p className="ls2">
                  -파기절차
                  <br />
                  이용자가 입력한 정보는 목적 달성 후 별도의 Database에 기록하고(종이의 경우 별도의 서류) 내부 방침 및 기타 관련 법령에 따라
                  일정기간 저장된 후 혹은 즉시 파기됩니다. 이 때, Database로 옮겨진 개인정보는 법률에 의한 경우가 아니고서는 다른 목적으로
                  이용되지 않습니다.
                  <br />
                  <br />
                  -파기기한
                  <br />
                  이용자의 개인정보는 개인정보의 보유기간이 경과된 경우에는 보유기간의 종료일로부터 5일 이내에, 개인정보의 처리 목적 달성,
                  해당 서비스의 폐지, 사업의 종료 등 그 개인정보가 불필요하게 되었을 때에는 개인정보의 처리가 불필요한 것으로 인정되는
                  날로부터 5일 이내에 그 개인정보를 파기합니다.
                </p>
                <p className="sub_p mgt10">-파기방법</p>
                <p className="sub_p">전자적 파일 형태의 정보는 기록을 재생할 수 없는 기술적 방법을 사용합니다.</p>
                <p>종이에 출력된 개인정보는 분쇄기로 분쇄하거나 소각을 통하여 파기합니다</p>
                <br />
                <p className="lh6 bs4">
                  <strong>6. 개인정보 자동 수집 장치의 설치•운영 및 거부에 관한 사항</strong>
                </p>
                <p className="ls2">
                  ① (주)웨다 은 개별적인 맞춤서비스를 제공하기 위해 이용정보를 저장하고 수시로 불러오는 ‘쿠기(cookie)’를 사용합니다.
                  <br /> ② 쿠키는 웹사이트를 운영하는데 이용되는 서버(http)가 이용자의 컴퓨터 브라우저에게 보내는 소량의 정보이며 이용자들의
                  PC 컴퓨터내의 하드디스크에 저장되기도 합니다.
                  <br /> 가. 쿠키의 사용 목적 : 이용자가 방문한 각 서비스와 웹 사이트들에 대한 방문 및 이용형태, 인기 검색어, 보안접속 여부,
                  등을 파악하여 이용자에게 최적화된 정보 제공을 위해 사용됩니다.
                  <br /> 나. 쿠키의 설치•운영 및 거부 : 웹브라우저 상단의 도구&gt;인터넷 옵션&gt;개인정보 메뉴의 옵션 설정을 통해 쿠키
                  저장을 거부 할 수 있습니다.
                  <br />
                  다. 쿠키 저장을 거부할 경우 맞춤형 서비스 이용에 어려움이 발생할 수 있습니다.
                </p>
                <br />
                <p className="sub_p mgt30">
                  <strong>7. 개인정보 보호책임자 작성 </strong>
                </p>
                <p className="sub_p mgt10">
                  ① <span className="colorLightBlue">{`(주)웨다(‘AI 비전 품질관리 솔루션’ 이하 ‘BluAi')`}</span> 은(는) 개인정보 처리에 관한
                  업무를 총괄해서 책임지고, 개인정보 처리와 관련한 정보주체의 불만처리 및 피해구제 등을 위하여 아래와 같이 개인정보
                  보호책임자를 지정하고 있습니다.
                </p>
                <ul className="list_indent2 mgt10"></ul>
                <ul className="list_indent2 mgt10">
                  <li className="tt">▶ 개인정보 보호 담당부서</li>
                  <li>부서명 :기술연구소</li>
                  <li>담당자 :BluAi 담당자</li>
                  <li>연락처 :02-6956-1017</li>
                  <li>메일 :info@weda.kr</li>
                </ul>
                <p className="sub_p">
                  ② 정보주체께서는 (주)웨다(‘AI 비전 품질관리 솔루션’ 이하 ‘BluAi’) 의 서비스(또는 사업)을 이용하시면서 발생한 모든 개인정보
                  보호 관련 문의, 불만처리, 피해구제 등에 관한 사항을 개인정보 보호책임자 및 담당부서로 문의하실 수 있습니다. (주)웨다(‘AI
                  비전 품질관리 솔루션’ 이하 ‘BluAi’) 은(는) 정보주체의 문의에 대해 지체 없이 답변 및 처리해드릴 것입니다.
                </p>
                <br />
                <p className="sub_p mgt30">
                  <strong>8. 개인정보 처리방침 변경 </strong>
                </p>
                <p className="sub_p mgt10">
                  ①이 개인정보처리방침은 시행일로부터 적용되며, 법령 및 방침에 따른 변경내용의 추가, 삭제 및 정정이 있는 경우에는 변경사항의
                  시행 7일 전부터 공지사항을 통하여 고지할 것입니다.
                </p>
                <br />
                <p className="lh6 bs4">
                  <strong>
                    9. 개인정보의 안전성 확보 조치 <em className="emphasis">{`<(주)웨다>('AI 비전 품질관리 솔루션')`}</em>은(는)
                    개인정보보호법 제29조에 따라 다음과 같이 안전성 확보에 필요한 기술적/관리적 및 물리적 조치를 하고 있습니다.
                  </strong>
                </p>
                <p className="sub_p mgt10">
                  1. 정기적인 자체 감사 실시
                  <br /> 개인정보 취급 관련 안정성 확보를 위해 정기적(분기 1회)으로 자체 감사를 실시하고 있습니다.
                  <br />
                  <br />
                  2. 개인정보 취급 직원의 최소화 및 교육
                  <br /> 개인정보를 취급하는 직원을 지정하고 담당자에 한정시켜 최소화 하여 개인정보를 관리하는 대책을 시행하고 있습니다.
                  <br />
                  <br />
                  3. 내부관리계획의 수립 및 시행
                  <br /> 개인정보의 안전한 처리를 위하여 내부관리계획을 수립하고 시행하고 있습니다.
                  <br />
                  <br />
                  4. 해킹 등에 대비한 기술적 대책
                  <br /> {`<`}
                  <em className="emphasis">(주)웨다</em>
                  {`>('`}
                  <em className="emphasis">AI 비전 품질관리 솔루션</em>
                  {`')은 해킹이나 컴퓨터 바이러스 등에 의한 개인정보 유출 및
                  훼손을 막기 위하여 보안프로그램을 설치하고 주기적인 갱신·점검을 하며 외부로부터 접근이 통제된 구역에 시스템을 설치하고
                  기술적/물리적으로 감시 및 차단하고 있습니다.`}
                  <br />
                  <br />
                  5. 접속기록의 보관 및 위변조 방지
                  <br /> 개인정보처리시스템에 접속한 기록을 최소 6개월 이상 보관, 관리하고 있으며, 접속 기록이 위변조 및 도난, 분실되지
                  않도록 보안기능 사용하고 있습니다.
                  <br />
                  <br />
                  6. 개인정보에 대한 접근 제한
                  <br /> 개인정보를 처리하는 데이터베이스시스템에 대한 접근권한의 부여,변경,말소를 통하여 개인정보에 대한 접근통제를 위하여
                  필요한 조치를 하고 있으며 침입차단시스템을 이용하여 외부로부터의 무단 접근을 통제하고 있습니다.
                  <br />
                  <br />
                  7. 문서보안을 위한 잠금장치 사용
                  <br /> 개인정보가 포함된 서류, 보조저장매체 등을 잠금장치가 있는 안전한 장소에 보관하고 있습니다.
                  <br />
                  <br />
                  8. 비인가자에 대한 출입 통제
                  <br /> 개인정보를 보관하고 있는 물리적 보관 장소를 별도로 두고 이에 대해 출입통제 절차를 수립, 운영하고 있습니다.
                  <br />
                  <br />
                </p>
              </p>
            </Col>
          </Row>
        </Container>
      </ModalBody>
    </Modal>
  )
}

PrivacyPolicyModal.propTypes = {}

export default PrivacyPolicyModal

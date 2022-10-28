import React, { useRef, useEffect } from "react";
import "./Kebab.css";

/* kebab 외 영역 감지 컴포넌ᄐᆕ */
function useOutsideAlerter(ref)
{
    // kebab 외 영역 클릭 이벤트
    function handleClickOutside(event)
    {
        if (ref.current && !ref.current.contains(event.target))
        {
            var dropdown = document.querySelectorAll(".dropdown");

            var activeCheck = false;

            dropdown.forEach((element) =>
            {
                if (element.classList.contains("active"))
                {
                    activeCheck = true;
                }
            });

            if (activeCheck && !event.target.className.includes("kebab") && !event.target.className.includes("dropdown"))
            {
                kebabRemove();

                event.stopPropagation();
                event.preventDefault();
            }
        }
    }

    /* kebab - 숨김 이벤트 */
    const kebabRemove = () =>
    {
        var dropdown = document.querySelectorAll(".dropdown");

        dropdown.forEach((element) =>
        {
            element.classList.remove("active");
        });
    };

    /* kebab 로드 시 호출 */
    useEffect(() => {
        document.addEventListener("mousedown", handleClickOutside);
        return () =>
        {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    });
}


function Kebab({ page, index, itemID, itemName, deleteItem, modifyItem, deleteAlter })
{
    /* Kebab - 클릭 이벤트 */
    const kebabClick = (index) =>
    {
        var dropdown = document.querySelectorAll(".dropdown");

        dropdown.forEach((element) =>
        {
            /* active 상태인 kebab 제거 (다른 kebab이 눌렸을 경우를 위해) */
            if (element.classList.contains("active") && dropdown[index] !== element)
            {
                element.classList.remove("active");
            }
        });
        dropdown[index].classList.toggle("active");
    };

    /* kebab item - 제거 버튼 클릭 이벤트 */
    const deleteHandler = (itemID, itemName) =>
    {
        if (window.confirm(`${deleteAlter} 삭제하시겠습니까 ?`) === true)
        {
            /* alert("삭제되었습니다"); */
            deleteItem(itemID, itemName);
        }
        else
        {
            return;
        }
    };

    /* kebab item - 이름 수정 버튼 클릭 이벤트 */
    const modifyHandler = (itemID, itemName) =>
    {
        modifyItem(itemID, itemName)
    };

  const wrapperRef = useRef(null);
  useOutsideAlerter(wrapperRef);

  return (
    <div className="kebab" ref={wrapperRef} onClick={(e) => { kebabClick(index); e.stopPropagation(); e.preventDefault(); }}>
      <figure className="kebab-figure"></figure>
      <figure className="kebab-figure"></figure>
      <figure className="kebab-figure"></figure>

        <div className="dropdown">

            {/* 현재 프로젝트 페이지에서 요청한 경우에만 표시 */}
            {page === 'project' &&
                <div className="kebab-modify" onClick={ () => modifyHandler(itemID, itemName) }>이름 수정</div>
            }

            {page === 'target' &&
                <div className="kebab-modify" onClick={ () => modifyHandler(itemID, itemName) }>수정</div>
            }

            <div className="kebab-delete" onClick={ () => deleteHandler(itemID, itemName) }>삭제</div>

        </div>


      </div>
  );

}

export default Kebab;

# How to PR (Pull Request)

References:
* [Git 협업 할 때 branch 생성 후 pull request까지의 과정](https://developer-eun-diary.tistory.com/42)

GitHub에 자신의 로컬브랜치(ex. `etri_feature01`)에서 작업한 결과물을 PR하기까지의 과정을 설명합니다.

* 실습이 필요한 경우, 본 프로젝트를 fork하여 자신의 GitHub 계정에 원격 저장소를 만들고, fork된 원격 저장소를 대상으로 아래의 내용을 실습하시면 본 프로젝트의 GitHub 저장소에 영향을 주지 않고 실습하실 수 있습니다.


## 용어 설명
* 원격 저장소(Remote Repo, Origin, Upstream등로 불림): https://github.com/hongsoog/DeepFramework 와 같이 공동으로 사용하는 저장소
* 로컬 저장소(Local Repo): 자신의 PC 또는 서버에 만든 git 저장소
* 브랜치(Brach): 특정 작업(릴리즈 관리, 디버그, 기능 추가, 기능 개선)을 위하여 생성하는 작업 흐름
   * master 브랜치: 기본 브랜치로 항상 빌드가능하고 실행가능한 코드를 유지하기 위한 브랜치
   * feature 브랜치: 특정 장업을 위하여 master브랜치에 영향을 주지않고 개발하기 위한 브랜치
   * 원격 브랜치: 원격 저장소의 브랜치
   * 로컬 브랜치: 로컬 저장소의 


## 1. 원격 저장소의 내용을 로컬 저장소로 가져오기

로컬 저장소가 없을 경우: `git clone`
```bash
$ git clone git@github.com:hongsoog/DeepFramework.git
```

로컬 저장소가 이미 있는 경우: `git pull origin master`
```bash
$ git pull origin master
```

## 2. 로컬저장소에서 자신만의 작업을 위한 로컬 브랜치 생성후 브랜치로 이동
```bash
# 작업용 로컬 브랜치(ex. etri_featuer01) 생성
$ git branch etri_featuer01

# 작업용 로컬 브랜치(ex. etri_featuer01)로 전환
$ git checkout etri_featuer01
```

위의 두개의 명령들은 다음과 같은 하나의 명령으로도 가능하다.
```bash
# 작업용 로컬 브랜치(ex. etri_featuer01) 생성후 바로 전환
$ git check out -b etri_featuer01
```

## 3. 작업용 로컬 브랜치에서 코드 수정 작성

* 현재 브랜치가 작업용 로컬 브랜치인지 확인할 것.
```bash
# 현재 브랜치 확인
$ git branch
  master
* etri_featuer01
```

* ```*``` 가 붙은 브랜치가 현재 브랜치임.


## 4. 작업용 로컬 브랜치의 작업 내용을 로컬 저장소에 커밋

현재 브랜치가 작업용 로컬 브랜치인지 확인할 것.

```bash
# 현재 브랜치 확인
$ git branch
  master
* etri_featuer01
```

* ```*``` 가 붙은 브랜치가 현재 브랜치임.

```bash
# Stage Area에 수정 파일 추가
$ git add .
sedan
# 작업 내용을 로컬 저장소에 커밋
$ git commit 
```

## 5. 로컬 저장소의 로컬 브랜치에서의 작업 내용을 원격 저장소에도 반영 push

원격 저장소에 로컬 작업용 브랜치가 존재하는 경우,
```bash
$ git push
```

원격 저장소에 로컬 작업용 브랜치가 존재하지 않는 경우,
```bash
$ git push origin etri_featuer01
```

* 원격 저장소에 원격 브랜치가 있으면 해당 브랜치 안에서 git push를 하면 바로 원격 저장소의 브랜치에 내용을 push 할 수 있다.
* 원격 저장소에 원격 브랜치가 없다면 git push만 했을 때, upstream이 없다는 오류가 나게 된다.
  * 원격 저장소에 브랜치 생성 후 브랜치 내용 push를 해줘야 한다.
  * 원격 브랜치의 이름은 혼돈을 주지 않도록, 가급적 로컬 브랜치와 동일한 이름을 사용한다.

## 6. 로컬 master 브랜치로 이동 후 원격 저장소의 원격 master 브랜치에 가해진 변경사항을 pull 

로컬 저장소에서 작업용 로컬 브랜치를 생성해서 작업을 하는 동안 다른 협업자가 원격 저장소의 master 원격 브랜치에 push를 진행해놓았을 수도 있다. 
그러므로 로컬 저장소의 master 로컬 브랜치로 이동 후, 원격 저장소의 master 원격 브랜치의 내용을 pull하여 로컬 저장소의 master 로컬 브랜치를 업데이트 해주어야 한다.

```bash
# master 로컬 브랜치로 전환
$ git checkout master

# 원격 저장소의 master 원격 브랜치의 최신 내용을 로컬 저장소의 master 로컬 브랜치에 반영
$ git pull origin master
```

## 7. 로컬 저장소의 작업용 로컬 브랜치(ex. etri_feature01)로 전환후 로컬 저장소의 master 로컬 브래친의 내용을 작업 브랜치(ex. etri_feature01)에 merge
```bash
# 로컬 작업 브랜치(ex. etri_feature01)로 전환
$ git checkout etri_feature01)

# 로컬 저장소의 master 로컬 브랜치의 최신 내용을 로컬 저장소의 작업용 로컬 브랜치(ex. etri_feature01)에 반영
$ git merge master 
```
merge 과정에서 conflict(코드의 동일한 위치를 2명 이상이 동시에 수정하여, Git에 의한 자동 merge가 불가능한 상황)가 발생하는 경우, 해결해주어야 한다.

git merge, conflict등에 대한 개념 설명은 아래의 생활코딩 유뷰트 영상을 참고
* [Hello Conflict - 알면 기능, 모르면 사고](https://youtu.be/wVUnsTsRQ3g)
* [3 Way Merge](https://youtu.be/J0W-WA0aYJI)
* [merge & conflict](https://youtu.be/0RqbZt_TZkY)


Conflict를 해결하고 나면, 로컬 저장소의 작업용 로컬 브랜치(ex. etri_feature01)의 작업 결과를 원격 저장소의 원격 브랜치(ex. etri_feature01)에 반영해준다.

```bash
# 작업 결과 원격 저장소 반영
$ git push

```

> 위의 과정을 통하여 master 로컬 브랜치를 merge한 작업용 로컬 브랜치(ex. etri_feature01)의 내용이 원격 저장소의 작업 브랜치(ㅕ. etri_feature01)에 업데이트되었다.

## 8. github 홈페이지의 repository로 이동 후 pull request 요청을 하면 완료


## 9. 담당자가 확인 후 pull request를 수락하면 원격 저장소 master에 브랜치의 내용이 업데이트된다

우리과제에서는 PR 요청하신 분이 직접 PR을 accept 하시면 됩니다.


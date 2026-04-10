import os
import json
from fastapi import APIRouter, Form, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Remote Sensing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 路由与鉴权配置 =================
# 创建一个专门处理认证和用户的路由器
auth_router = APIRouter()
# 声明 OAuth2 规则，指定获取 token 的接口路径为 /token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ================= 简易本地 JSON 用户“数据库” =================
USER_FILE = r"D:\RuanZhu-wc-v2\user.json"

if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump([{"id": 1, "username": "admin", "password": "123456"}], f)

def get_all_users():
    with open(USER_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_all_users(users):
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

class UserBase(BaseModel):
    username: str
    password: str

class ResetPassword(BaseModel):
    username: str
    new_password: str

# ================= 用户增删改查及登录 API =================

@auth_router.post("/token")
async def login(username: str = Form(...), password: str = Form(...)):
    users = get_all_users()
    for u in users:
        if u["username"] == username and u["password"] == password:
            return {"access_token": f"token_for_{username}", "token_type": "bearer"}
    return {"error": "账号或密码错误"}

# [新增] 公开的注册接口（无需鉴权）
@auth_router.post("/register")
async def register_user_public(user: UserBase):
    users = get_all_users()
    if any(u["username"] == user.username for u in users):
        return {"success": False, "message": "用户名已存在，请换一个"}
    
    new_id = max([u["id"] for u in users], default=0) + 1
    users.append({"id": new_id, "username": user.username, "password": user.password})
    save_all_users(users)
    return {"success": True, "message": "注册成功，请返回登录"}

# [新增] 忘记密码的重置接口（无需鉴权）
@auth_router.post("/reset_password")
async def reset_password(data: ResetPassword):
    users = get_all_users()
    for i, u in enumerate(users):
        if u["username"] == data.username:
            users[i]["password"] = data.new_password
            save_all_users(users)
            return {"success": True, "message": "密码重置成功，请使用新密码登录"}
    return {"success": False, "message": "找不到该用户账号"}

@auth_router.get("/users")
async def read_users(token: str = Depends(oauth2_scheme)):
    return {"success": True, "data": get_all_users()}

@auth_router.post("/users")
async def create_user(user: UserBase, token: str = Depends(oauth2_scheme)):
    users = get_all_users()
    if any(u["username"] == user.username for u in users):
        return {"success": False, "message": "用户名已存在，请换一个"}
    
    new_id = max([u["id"] for u in users], default=0) + 1
    users.append({"id": new_id, "username": user.username, "password": user.password})
    save_all_users(users)
    return {"success": True, "message": "用户添加成功"}

@auth_router.put("/users/{user_id}")
async def update_user(user_id: int, user: UserBase, token: str = Depends(oauth2_scheme)):
    users = get_all_users()
    for i, u in enumerate(users):
        if u["id"] == user_id:
            if user.username != u["username"] and any(x["username"] == user.username for x in users):
                return {"success": False, "message": "该用户名已被别人使用"}
            users[i]["username"] = user.username
            users[i]["password"] = user.password
            save_all_users(users)
            return {"success": True, "message": "用户信息更新成功"}
    return {"success": False, "message": "找不到该用户"}

@auth_router.delete("/users/{user_id}")
async def delete_user(user_id: int, token: str = Depends(oauth2_scheme)):
    users = get_all_users()
    if len(users) <= 1:
        return {"success": False, "message": "系统中至少需要保留一个账号！"}
    
    users = [u for u in users if u["id"] != user_id]
    save_all_users(users)
    return {"success": True, "message": "用户删除成功"}

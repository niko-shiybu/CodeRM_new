# Docker 设置指南

## Docker 的作用

在这个实验中，**Docker 用于安全地执行生成的代码和单元测试**。

### 为什么需要 Docker？

1. **安全性**：
   - 生成的代码可能包含恶意代码（删除文件、访问系统资源等）
   - Docker 提供隔离的执行环境，防止代码影响主机系统

2. **环境一致性**：
   - 确保所有代码在相同的 Python 环境中执行
   - 避免因环境差异导致的执行结果不一致

3. **资源限制**：
   - 可以限制执行时间和内存使用
   - 防止代码无限循环或占用过多资源

### Docker 在实验中的使用

在 `evaluation/evaluate.py` 中，Docker 用于：
- 执行每个解决方案 + 单元测试的组合
- 捕获执行结果（pass/fail/error）
- 批量处理大量测试用例（支持多进程并行）

## 权限问题解决方案

你遇到的错误：
```
permission denied while trying to connect to the Docker daemon socket
```

这是因为你的用户没有权限访问 Docker daemon。有几种解决方法：

### 方法 1：将用户添加到 docker 组（推荐）

这是最常用的方法，可以让你的用户无需 `sudo` 就能使用 Docker：

```bash
# 1. 将当前用户添加到 docker 组
sudo usermod -aG docker $USER

# 2. 重新登录或刷新组权限
# 方法 A: 重新登录（推荐）
# 退出当前会话，重新 SSH 登录

# 方法 B: 使用 newgrp（临时生效）
newgrp docker

# 3. 验证权限
docker ps
# 如果能看到容器列表（即使为空），说明权限已生效

# 4. 拉取镜像
docker pull kaka0605/exec_unit_test:24.12.30
```

**注意**：如果是在服务器上，可能需要联系管理员执行 `sudo usermod` 命令。

### 方法 2：使用 sudo（临时方案）

如果无法添加到 docker 组，可以使用 `sudo`：

```bash
# 拉取镜像
sudo docker pull kaka0605/exec_unit_test:24.12.30

# 但这样每次运行都需要 sudo，不太方便
```

如果使用这个方法，需要修改 `evaluation/evaluate.py` 中的 Docker 命令，在所有 `docker` 命令前加上 `sudo`。

### 方法 3：检查 Docker 服务状态

如果 Docker 服务没有运行，也会出现权限错误：

```bash
# 检查 Docker 服务状态
sudo systemctl status docker

# 如果服务未运行，启动它
sudo systemctl start docker

# 设置开机自启
sudo systemctl enable docker
```

### 方法 4：使用 Docker Desktop（如果服务器支持 GUI）

如果服务器有图形界面，可以安装 Docker Desktop，它会自动处理权限问题。

## 验证 Docker 设置

完成权限配置后，运行以下命令验证：

```bash
# 1. 检查 Docker 版本
docker --version

# 2. 检查 Docker 服务状态
docker ps

# 3. 拉取实验所需的镜像
docker pull kaka0605/exec_unit_test:24.12.30

# 4. 验证镜像是否拉取成功
docker images | grep exec_unit_test

# 5. 测试运行（可选）
docker run --rm kaka0605/exec_unit_test:24.12.30 --help
```

## 如果无法使用 Docker

如果服务器上无法使用 Docker（例如没有 root 权限），可以考虑以下替代方案：

### 方案 A：使用 Python 虚拟环境（不推荐，安全性较低）

修改 `evaluation/evaluate.py`，直接使用 Python 执行代码而不是 Docker。但这会降低安全性。

### 方案 B：使用其他容器技术

- **Podman**：无需 root 权限的容器工具
- **Singularity**：常用于 HPC 环境的容器工具

### 方案 C：使用云服务

使用支持 Docker 的云服务器（AWS、GCP、Azure 等）运行实验。

## 修改脚本以支持 sudo（如果需要）

如果你必须使用 `sudo`，可以修改 `evaluation/evaluate.py`：

```python
# 修改第 8 行的 UT_EXEC_FORMAT
UT_EXEC_FORMAT = """sudo docker run -v $(pwd):/data kaka0605/exec_unit_test:24.12.30 \
    --input_path /data/output/{benchmark}/{sol_model}_sol_{ut_model}_ut/details/sol_ut.jsonl \
    --output_path /data/{docker_write}/{sol_num}_sol_{ut_num}_ut_result.jsonl \
    --mp_num {mp_num} \
    --chunk_size 1000 \
    --recover 0
"""
```

**注意**：使用 `sudo` 需要配置免密 sudo，否则每次执行都需要输入密码。

## 常见问题

### Q1: 添加用户到 docker 组后仍然报错？

**A**: 需要重新登录 SSH 会话，或者运行 `newgrp docker`。

### Q2: 服务器管理员不允许添加用户到 docker 组？

**A**: 
- 使用 `sudo docker`（需要配置免密 sudo）
- 或者联系管理员说明实验需求

### Q3: Docker 镜像拉取很慢？

**A**: 可以配置 Docker 镜像加速器（国内服务器）：

```bash
# 编辑 Docker daemon 配置
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com"
  ]
}
EOF

# 重启 Docker 服务
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### Q4: 如何检查 Docker 是否正常工作？

**A**: 运行测试命令：

```bash
docker run hello-world
```

如果能看到 "Hello from Docker!" 消息，说明 Docker 正常工作。

## 总结

1. **Docker 的作用**：安全执行代码和单元测试
2. **权限问题**：将用户添加到 docker 组（推荐）或使用 sudo
3. **验证**：运行 `docker ps` 和 `docker pull` 测试
4. **如果无法使用 Docker**：考虑使用云服务器或其他容器技术

如果遇到其他问题，请检查：
- Docker 服务是否运行：`sudo systemctl status docker`
- 用户是否在 docker 组：`groups $USER`
- Docker socket 权限：`ls -l /var/run/docker.sock`

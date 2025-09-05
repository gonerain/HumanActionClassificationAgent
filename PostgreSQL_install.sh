#!/usr/bin/env bash
set -e

# ====== 基础准备 ======
echo "[1/6] 安装依赖工具..."
sudo apt-get update -y
sudo apt-get install -y wget gnupg lsb-release ca-certificates

# ====== 安装 PostgreSQL 14 ======
echo "[2/6] 安装 PostgreSQL 14..."
sudo apt-get install -y postgresql-14 postgresql-client-14

# ====== 添加 TimescaleDB 仓库 ======
echo "[3/6] 添加 TimescaleDB 官方仓库..."
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/timescaledb.gpg

echo "deb [signed-by=/usr/share/keyrings/timescaledb.gpg] \
https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/timescaledb.list

sudo apt-get update -y

# ====== 安装 TimescaleDB for PG14 ======
echo "[4/6] 安装 TimescaleDB (PostgreSQL 14 版本)..."
sudo apt-get install -y timescaledb-2-postgresql-14

# ====== 调优配置 ======
echo "[5/6] 配置 PostgreSQL 以启用 TimescaleDB..."
sudo timescaledb-tune --yes

# ====== 重启 PostgreSQL ======
echo "[6/6] 重启 PostgreSQL 服务..."
sudo systemctl restart postgresql

echo "✅ PostgreSQL 14 + TimescaleDB 已安装完成"
echo "请执行以下命令在数据库里启用扩展："
echo "    sudo -u postgres psql -d mydb -c \"CREATE EXTENSION IF NOT EXISTS timescaledb;\""

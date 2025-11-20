# Setting Up MongoDB on WSL

**Prerequisites:**

- Windows 10/11 with WSL2 installed
- Ubuntu or Debian distribution in WSL
- VS Code installed (optional)

***

## 1. Update System

Update package lists to ensure latest versions and avoid compatibility issues.

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

***

## 2. Import MongoDB Repository Key

MongoDB isn't in Ubuntu's default repositories. Import MongoDB's official GPG key to verify authenticity.

```bash
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
```

***

## 3. Add MongoDB Repository

Tell Ubuntu's package manager where to find MongoDB packages.

```bash
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | \
   sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

sudo apt-get update
```

***

## 4. Install MongoDB

Install the complete MongoDB database system including server, shell, and tools.

```bash
sudo apt-get install -y mongodb-org
```

Verify installation:

```bash
mongod --version
mongosh --version
```

***

## 5. Create Data Directory

MongoDB requires a directory to store database files. Default location is `/data/db`.

```bash
# Create directory
sudo mkdir -p /data/db

# Set ownership to your user
sudo chown -R $USER:$USER /data/db

# Verify permissions
ls -ld /data/db
```

Expected output: `drwxr-xr-x ... /data/db`

***

## 6. Start MongoDB Server

**Method 1: Using systemctl**

```bash
# Start MongoDB service
sudo systemctl start mongod

# Check status
sudo systemctl status mongod

# Enable auto-start on boot (optional)
sudo systemctl enable mongod
```

Look for "Active: active (running)" in status output.

**Method 2: Manual start (if systemctl doesn't work)**

```bash
sudo mongod --dbpath /data/db --fork --logpath /var/log/mongodb.log
```

Flags:
- `--dbpath /data/db` - Data storage location
- `--fork` - Runs in background
- `--logpath` - Log file location

***

## 7. Connect to MongoDB Shell

Launch MongoDB shell to interact with databases.

```bash
mongosh
```

Expected output:
```
Connecting to: mongodb://127.0.0.1:27017/
Using MongoDB: 7.0.x
test>
```

The `test>` prompt indicates connection success.

***

## 8. Test MongoDB Operations

Run these commands inside mongosh:

**Check current database:**
```javascript
db
```

**Create database:**
```javascript
use testdb
```

**Insert document:**
```javascript
db.users.insertOne({ 
  name: "Alice", 
  age: 28, 
  city: "Tirana"
})
```

**Query documents:**
```javascript
db.users.find()
```

**Insert multiple documents:**
```javascript
db.users.insertMany([
  { name: "Bob", age: 32, city: "Durrës" },
  { name: "Carol", age: 25, city: "Vlorë" }
])
```

**Count documents:**
```javascript
db.users.countDocuments()
```

**Query with filter:**
```javascript
db.users.find({ age: { $gte: 30 } })
```

**Update document:**
```javascript
db.users.updateOne(
  { name: "Alice" },
  { $set: { age: 29 } }
)
```

**Delete document:**
```javascript
db.users.deleteOne({ name: "Bob" })
```

**Show collections:**
```javascript
show collections
```

**Show databases:**
```javascript
show dbs
```

**Exit shell:**
```javascript
exit
```

***

## 9. Stop MongoDB Server

```bash
sudo systemctl stop mongod
```

Or if started manually:
```bash
sudo pkill mongod
```

***

## 10. VS Code Integration (Optional)

**Install extension:**

1. Open VS Code
2. Extensions (Ctrl+Shift+X)
3. Search "MongoDB for VS Code"
4. Install official extension

**Connect:**

1. Click MongoDB icon in sidebar
2. Add Connection
3. Enter: `mongodb://localhost:27017`
4. Connect

Features: Browse databases, run queries, view documents, export/import data.


## Verification Commands

```bash
# Check versions
mongod --version
mongosh --version

# Verify service
sudo systemctl status mongod

# Test connection
mongosh --eval "db.version()"
```


## Essential Commands

```bash
# Start
sudo systemctl start mongod

# Connect
mongosh

# Stop
sudo systemctl stop mongod

# Status
sudo systemctl status mongod
```

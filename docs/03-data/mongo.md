# MongoDB Introduction: NoSQL Document Databases

This tutorial introduces MongoDB and NoSQL databases. You'll learn what NoSQL means, how it differs from SQL, when to choose document storage, and how to perform CRUD operations (Create, Read, Update, Delete) in MongoDB using practical examples.

**Estimated time:** 45 minutes

## Why This Matters

**Problem statement:** 

> Not all data fits neatly into tables.

**Real-world data is messy**. Social media posts with varying numbers of tags, product catalogs where items have different attributes, IoT sensors generating unpredictable JSON payloads, and user profiles with flexible schemas don't map cleanly to rigid SQL tables. Adding columns for every possible field creates sparse tables full of NULLs.

**NoSQL databases solve these problems**. They embrace flexibility, allowing each record to have its own structure. **MongoDB stores data as JSON-like documents**, making it natural to work with modern web applications that already speak JSON.

**Practical benefits:** MongoDB skills let you build applications with evolving schemas, handle unstructured data from APIs, scale horizontally across servers, and integrate seamlessly with JavaScript/Python ecosystems. Many startups and tech companies choose MongoDB for rapid prototyping and flexible data models.

**Professional context:** NoSQL databases power high-traffic applications like e-commerce catalogs, content management systems, real-time analytics, and mobile app backends. Understanding when to use NoSQL vs SQL is a crucial architectural decision.

> Choose the right tool for the job—SQL for structured transactions, NoSQL for flexible documents.

## Core Concepts

### What Is NoSQL?

**NoSQL = "Not Only SQL"** (not "No SQL")

NoSQL databases provide alternative data models to traditional relational databases. They sacrifice some relational features (like complex joins and strict schemas) to gain flexibility, scalability, and performance for specific use cases.

**Key characteristics:**

- **Schema flexibility:** Documents can have different structures
- **Horizontal scaling:** Add more servers instead of upgrading hardware
- **Denormalization:** Store related data together rather than splitting across tables
- **High availability:** Built-in replication and distribution

**Common misconception:** NoSQL doesn't mean "no structure"; it means "flexible structure."

### SQL vs NoSQL: Key Differences

| Aspect | SQL (Relational) | NoSQL (MongoDB) |
|--------|------------------|-----------------|
| **Data Model** | Tables with rows/columns | Collections with documents (JSON) |
| **Schema** | Fixed, predefined | Dynamic, flexible |
| **Relationships** | Foreign keys, JOINs | Embedded documents or references |
| **Scaling** | Vertical (bigger servers) | Horizontal (more servers) |
| **Transactions** | Strong ACID guarantees | Eventual consistency (varies) |
| **Query Language** | SQL | Query API (JavaScript-like) |
| **Best For** | Financial systems, structured data | Content management, catalogs, logs |

**Example comparison:**

**SQL structure:**
```sql
-- Two tables with foreign key
Customers: id, name, email
Orders: id, customer_id, order_date, amount
```

**NoSQL structure:**
```javascript
// Single document with embedded data
{
  "_id": "cust123",
  "name": "John Doe",
  "email": "john@example.com",
  "orders": [
    { "order_date": "2025-01-15", "amount": 99.99 },
    { "order_date": "2025-02-20", "amount": 149.50 }
  ]
}
```

### What Is ACID?

**ACID = Atomicity, Consistency, Isolation, Durability**

These properties guarantee reliable database transactions:

1. **Atomicity:** All operations in a transaction succeed or all fail (no partial updates)
2. **Consistency:** Database moves from one valid state to another (constraints maintained)
3. **Isolation:** Concurrent transactions don't interfere with each other
4. **Durability:** Committed data survives system failures

**SQL databases:** Strong ACID by default (every operation is a transaction)

**NoSQL databases (including MongoDB):** 

- Traditionally prioritized performance over strict ACID
- MongoDB 4.0+ supports multi-document ACID transactions
- Single-document operations are always atomic

**Trade-off:** Traditional NoSQL favored eventual consistency for speed and scalability. Modern versions like MongoDB now offer both options.

### What Is Document Storage?

**Document databases** store data as documents (JSON, BSON, XML). Each document is a self-contained record with key-value pairs.

**Example document:**
```javascript
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "name": "Alice Johnson",
  "email": "alice@company.com",
  "age": 28,
  "skills": ["Python", "SQL", "Docker"],
  "address": {
    "city": "Tirana",
    "country": "Albania"
  },
  "hire_date": ISODate("2025-01-15T00:00:00Z"),
  "active": true
}
```

**Why documents?**

- **Natural fit for objects:** Maps directly to JSON/Python dictionaries
- **No schema migration:** Add fields without ALTER TABLE
- **Nested structures:** Embed related data (address, arrays)
- **Variable fields:** Different documents can have different fields

**MongoDB uses BSON** (Binary JSON) internally for efficiency while exposing JSON interface.

### NoSQL Types

NoSQL encompasses multiple database types, each optimized for different use cases:

**1. Document Databases** (MongoDB, CouchDB)

- Store JSON-like documents
- Best for: Content management, catalogs, user profiles

**2. Key-Value Stores** (Redis, DynamoDB)

- Simple key → value mapping
- Best for: Caching, session storage, real-time data

**3. Column-Family Stores** (Cassandra, HBase)

- Store data in columns rather than rows
- Best for: Time-series data, analytics, massive scale

**4. Graph Databases** (Neo4j, Amazon Neptune)

- Store nodes and relationships
- Best for: Social networks, recommendation engines, fraud detection

**This tutorial focuses on MongoDB:** The most popular document database.

### Benefits of NoSQL Databases

**Flexibility:** Add fields to documents without downtime or migrations. Perfect for evolving requirements.

```javascript
// No problem adding new field to one document
{ "name": "Bob", "email": "bob@co.com", "department": "Sales" }
```


**When to choose NoSQL:**

- Rapid development with changing requirements
- Hierarchical or nested data structures
- Need to scale horizontally
- Working with JSON APIs
- Real-time analytics on large datasets

**When to stick with SQL:**

- Complex transactions (banking, accounting)
- Strong consistency requirements
- Data with many relationships requiring joins
- Existing SQL ecosystem and expertise

## Step-by-Step Guide

### 1. MongoDB Basics and Setup

**Installing MongoDB:**

```bash
# macOS with Homebrew
brew tap mongodb/brew
brew install mongodb-community

# Ubuntu/Debian
sudo apt-get install mongodb

# Windows: Download installer from mongodb.com
```

**Starting MongoDB:**

```bash
# Start MongoDB service
mongod

# Connect with MongoDB Shell
mongosh
```

**Cloud alternative:** Use MongoDB Atlas (free tier available) at [mongodb.com/atlas](https://www.mongodb.com/atlas)

**Basic MongoDB terminology:**

| SQL Term | MongoDB Equivalent |
|----------|-------------------|
| Database | Database |
| Table | Collection |
| Row | Document |
| Column | Field |
| Index | Index |
| JOIN | Embedding or $lookup |

### 2. Creating Databases and Collections

**In MongoDB, databases and collections are created automatically when you insert data.**

**Switch to (or create) database:**

```javascript
// Switch to database (creates if doesn't exist)
use company_data

// Check current database
db

// Show all databases
show dbs
// Note: Empty databases don't appear until data is added
```

**Collections are created implicitly:**

```javascript
// No need to explicitly create collection
// It's created automatically on first insert
db.employees.insertOne({
  name: "John Doe",
  email: "john@company.com"
})

// Show all collections in current database
show collections

// Explicitly create collection (optional)
db.createCollection("customers")
```

**Dropping database/collection:**

```javascript
// Drop current database
db.dropDatabase()

// Drop collection
db.employees.drop()
```

### 3. Inserting Documents

**Insert single document:**

```javascript
// insertOne() adds a single document
db.employees.insertOne({
  first_name: "Alice",
  last_name: "Johnson",
  email: "alice.j@company.com",
  age: 28,
  department: "Engineering",
  skills: ["Python", "MongoDB", "Docker"],
  hire_date: new Date("2025-01-15"),
  salary: 75000,
  active: true
})

// MongoDB automatically generates _id if not provided
// Returns: { acknowledged: true, insertedId: ObjectId("...") }
```

**Insert multiple documents:**

```javascript
// insertMany() adds array of documents
db.employees.insertMany([
  {
    first_name: "Bob",
    last_name: "Smith",
    email: "bob.smith@company.com",
    age: 35,
    department: "Sales",
    skills: ["Negotiation", "CRM"],
    hire_date: new Date("2024-06-10"),
    salary: 68000,
    active: true
  },
  {
    first_name: "Carol",
    last_name: "Martinez",
    email: "carol.m@company.com",
    age: 42,
    department: "Engineering",
    skills: ["JavaScript", "React", "Node.js"],
    hire_date: new Date("2023-03-22"),
    salary: 92000,
    active: true
  },
  {
    first_name: "David",
    last_name: "Chen",
    email: "david.chen@company.com",
    age: 29,
    department: "Marketing",
    skills: ["SEO", "Content", "Analytics"],
    hire_date: new Date("2024-11-05"),
    salary: 62000,
    active: true
  }
])

// Returns: { acknowledged: true, insertedIds: { '0': ObjectId("..."), '1': ObjectId("..."), ... } }
```

**Nested documents:**

```javascript
// Documents can contain nested objects and arrays
db.employees.insertOne({
  first_name: "Emma",
  last_name: "Wilson",
  email: "emma.w@company.com",
  age: 31,
  department: "HR",
  address: {
    street: "123 Main St",
    city: "Tirana",
    country: "Albania",
    postal_code: "1001"
  },
  projects: [
    { name: "Recruitment Portal", role: "Lead", start_date: new Date("2024-01-01") },
    { name: "Training Program", role: "Coordinator", start_date: new Date("2024-06-15") }
  ],
  salary: 70000
})
```

### 4. Querying Documents

**MongoDB uses a rich query API instead of SQL.**

**Find all documents:**

```javascript
// Find all (like SELECT *)
db.employees.find()

// Pretty print
db.employees.find().pretty()

// Count documents
db.employees.countDocuments()
```

**Find with criteria (WHERE equivalent):**

```javascript
// Single condition
db.employees.find({ department: "Engineering" })

// Multiple conditions (AND)
db.employees.find({ 
  department: "Engineering", 
  salary: { $gte: 80000 } 
})

// OR condition
db.employees.find({
  $or: [
    { department: "Engineering" },
    { department: "Sales" }
  ]
})

// IN operator
db.employees.find({
  department: { $in: ["Engineering", "Sales", "Marketing"] }
})
```

**Comparison operators:**

| Operator | Meaning | Example |
|----------|---------|---------|
| `$eq` | Equals | `{ age: { $eq: 30 } }` |
| `$ne` | Not equals | `{ active: { $ne: false } }` |
| `$gt` | Greater than | `{ salary: { $gt: 70000 } }` |
| `$gte` | Greater or equal | `{ age: { $gte: 30 } }` |
| `$lt` | Less than | `{ age: { $lt: 40 } }` |
| `$lte` | Less or equal | `{ salary: { $lte: 80000 } }` |
| `$in` | In array | `{ dept: { $in: ["IT", "HR"] } }` |
| `$nin` | Not in array | `{ status: { $nin: ["inactive"] } }` |

**Projection (select specific fields):**

```javascript
// Show only first_name and email (1 = include, 0 = exclude)
db.employees.find(
  { department: "Engineering" },
  { first_name: 1, email: 1, _id: 0 }
)

// Exclude specific fields
db.employees.find(
  {},
  { salary: 0, _id: 0 }
)
```

**Sorting:**

```javascript
// Sort by salary ascending (1 = ascending, -1 = descending)
db.employees.find().sort({ salary: 1 })

// Sort by department ascending, then salary descending
db.employees.find().sort({ department: 1, salary: -1 })
```

**Limiting and skipping:**

```javascript
// Get first 5 documents
db.employees.find().limit(5)

// Skip first 10, then get 5 (pagination)
db.employees.find().skip(10).limit(5)

// Combine: sort, skip, limit
db.employees.find()
  .sort({ hire_date: -1 })
  .skip(10)
  .limit(5)
```

**Pattern matching (LIKE equivalent):**

```javascript
// Regex for pattern matching
// Find emails ending with @company.com
db.employees.find({
  email: { $regex: "@company.com$" }
})

// Case-insensitive search
db.employees.find({
  first_name: { $regex: "^a", $options: "i" }  // Starts with 'a' or 'A'
})
```

**Querying nested fields:**

```javascript
// Dot notation for nested fields
db.employees.find({
  "address.city": "Tirana"
})

// Query array elements
db.employees.find({
  skills: "Python"  // Finds if Python is in skills array
})

// Array contains all
db.employees.find({
  skills: { $all: ["Python", "MongoDB"] }
})
```

**Find one document:**

```javascript
// Returns single document (or null)
db.employees.findOne({ email: "alice.j@company.com" })

// Useful for getting by ID
db.employees.findOne({ _id: ObjectId("507f1f77bcf86cd799439011") })
```

### 5. Updating Documents

**Update single document:**

```javascript
// updateOne() modifies first matching document
db.employees.updateOne(
  { email: "alice.j@company.com" },  // Filter
  { $set: { salary: 82000 } }        // Update
)

// Returns: { acknowledged: true, matchedCount: 1, modifiedCount: 1 }
```

**Update operators:**

| Operator | Purpose | Example |
|----------|---------|---------|
| `$set` | Set field value | `{ $set: { age: 30 } }` |
| `$unset` | Remove field | `{ $unset: { temp_field: "" } }` |
| `$inc` | Increment number | `{ $inc: { salary: 5000 } }` |
| `$mul` | Multiply | `{ $mul: { quantity: 2 } }` |
| `$rename` | Rename field | `{ $rename: { "name": "full_name" } }` |
| `$push` | Add to array | `{ $push: { skills: "Docker" } }` |
| `$pull` | Remove from array | `{ $pull: { skills: "Java" } }` |
| `$addToSet` | Add if not exists | `{ $addToSet: { tags: "new" } }` |

**Update multiple documents:**

```javascript
// updateMany() modifies all matching documents
db.employees.updateMany(
  { department: "Engineering" },
  { $inc: { salary: 5000 } }  // Give 5k raise to all engineers
)

// Update all documents
db.employees.updateMany(
  {},
  { $set: { reviewed: false } }
)
```

**Update with multiple operators:**

```javascript
db.employees.updateOne(
  { email: "bob.smith@company.com" },
  {
    $set: { department: "Sales Management" },
    $inc: { salary: 10000 },
    $push: { skills: "Leadership" }
  }
)
```

**Upsert (update or insert):**

```javascript
// If document doesn't exist, create it
db.employees.updateOne(
  { email: "new.person@company.com" },
  { 
    $set: { 
      first_name: "New",
      last_name: "Person",
      department: "IT",
      salary: 60000
    }
  },
  { upsert: true }  // Creates if not found
)
```

**Replace entire document:**

```javascript
// replaceOne() replaces entire document (except _id)
db.employees.replaceOne(
  { email: "old@company.com" },
  {
    first_name: "Updated",
    last_name: "Employee",
    email: "new@company.com",
    department: "Finance",
    salary: 75000
  }
)
// Warning: This removes all fields not in replacement document
```

### 6. Deleting Documents

**Delete single document:**

```javascript
// deleteOne() removes first matching document
db.employees.deleteOne({
  email: "person@company.com"
})

// Returns: { acknowledged: true, deletedCount: 1 }
```

**Delete multiple documents:**

```javascript
// deleteMany() removes all matching documents
db.employees.deleteMany({
  active: false
})

// Delete all documents in collection
db.employees.deleteMany({})  // Dangerous!
```

**Delete with conditions:**

```javascript
// Delete employees hired before 2023
db.employees.deleteMany({
  hire_date: { $lt: new Date("2023-01-01") }
})

// Delete by multiple criteria
db.employees.deleteMany({
  department: "Temp",
  salary: { $lt: 50000 }
})
```

### 7. Aggregation Pipeline

**Aggregation performs complex data processing (like SQL GROUP BY, JOINs).**

**Basic aggregation:**

```javascript
// Calculate average salary by department
db.employees.aggregate([
  {
    $group: {
      _id: "$department",
      avg_salary: { $avg: "$salary" },
      count: { $sum: 1 }
    }
  },
  {
    $sort: { avg_salary: -1 }
  }
])
```

**Aggregation stages:**

| Stage | Purpose | SQL Equivalent |
|-------|---------|----------------|
| `$match` | Filter documents | WHERE |
| `$group` | Group by field | GROUP BY |
| `$sort` | Sort results | ORDER BY |
| `$project` | Select/reshape fields | SELECT |
| `$limit` | Limit results | LIMIT |
| `$skip` | Skip documents | OFFSET |
| `$lookup` | Join collections | JOIN |
| `$unwind` | Deconstruct arrays | - |

**Complex aggregation example:**

```javascript
// Find top 3 highest paid employees per department
db.employees.aggregate([
  // Stage 1: Filter active employees
  {
    $match: { active: true }
  },
  // Stage 2: Sort by department and salary
  {
    $sort: { department: 1, salary: -1 }
  },
  // Stage 3: Group by department and get top 3
  {
    $group: {
      _id: "$department",
      top_earners: { 
        $push: {
          name: { $concat: ["$first_name", " ", "$last_name"] },
          salary: "$salary"
        }
      }
    }
  },
  // Stage 4: Limit to top 3 per department
  {
    $project: {
      department: "$_id",
      top_earners: { $slice: ["$top_earners", 3] }
    }
  }
])
```

**Common aggregation operators:**

```javascript
// Count, sum, average, min, max
db.employees.aggregate([
  {
    $group: {
      _id: null,  // Group all documents together
      total_employees: { $sum: 1 },
      total_payroll: { $sum: "$salary" },
      avg_salary: { $avg: "$salary" },
      min_salary: { $min: "$salary" },
      max_salary: { $max: "$salary" }
    }
  }
])
```


## Common MongoDB Challenges

### Handling Missing Fields

**Problem:** Documents can have different fields. Querying missing fields returns nothing.

```javascript
// Check if field exists
db.employees.find({ phone: { $exists: true } })

// Check if field doesn't exist
db.employees.find({ phone: { $exists: false } })

// Provide default with $ifNull in aggregation
db.employees.aggregate([
  {
    $project: {
      name: "$first_name",
      phone: { $ifNull: ["$phone", "No phone provided"] }
    }
  }
])
```

### Understanding _id vs Custom IDs

**Problem:** MongoDB auto-generates `_id` as ObjectId, but sometimes you want custom IDs.

```javascript
// Auto-generated ObjectId
db.users.insertOne({ name: "Alice" })
// _id: ObjectId("507f1f77bcf86cd799439011")

// Custom ID
db.users.insertOne({ _id: "user_123", name: "Bob" })
// _id: "user_123"

// Query by ObjectId (must wrap in ObjectId())
db.users.findOne({ _id: ObjectId("507f1f77bcf86cd799439011") })

// Query by custom ID
db.users.findOne({ _id: "user_123" })
```

### Avoiding Accidental Updates

**Problem:** Forgetting `$set` replaces entire document.

```javascript
// WRONG: This replaces entire document with just { salary: 80000 }
db.employees.updateOne(
  { email: "alice@co.com" },
  { salary: 80000 }  // Missing $set!
)

// CORRECT: Use $set to update specific fields
db.employees.updateOne(
  { email: "alice@co.com" },
  { $set: { salary: 80000 } }
)
```

## Quick Reference

### Essential MongoDB Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `use <db>` | Switch/create database | `use company` |
| `show dbs` | List databases | `show dbs` |
| `show collections` | List collections | `show collections` |
| `db.collection.insertOne()` | Insert document | `db.users.insertOne({name: "Alice"})` |
| `db.collection.find()` | Query documents | `db.users.find({age: {$gt: 25}})` |
| `db.collection.updateOne()` | Update document | `db.users.updateOne({_id: 1}, {$set: {age: 30}})` |
| `db.collection.deleteOne()` | Delete document | `db.users.deleteOne({_id: 1})` |
| `db.collection.countDocuments()` | Count documents | `db.users.countDocuments()` |
| `db.collection.drop()` | Delete collection | `db.users.drop()` |


## Summary & Next Steps

**Key accomplishments:** You've learned what NoSQL means and when to use it, how MongoDB differs from SQL databases, what ACID properties ensure, how document storage works with JSON-like structures, the four types of NoSQL databases, how to perform CRUD operations in MongoDB, and how to use aggregation for complex queries.

**Critical insights:**

- **NoSQL isn't anti-SQL:** It's a complementary tool for different use cases
- **Flexibility has trade-offs:** Gain schema freedom, lose strict consistency guarantees
- **Denormalization is normal:** Embed related data instead of always splitting into references
- **Indexes matter:** Even flexible databases need optimization

**When to use MongoDB:**

- Rapid prototyping with evolving requirements
- Hierarchical/nested data (product catalogs, user profiles)
- Content management systems
- Real-time analytics and logging
- Applications already using JSON

**When to use SQL:**

- Financial transactions requiring strong consistency
- Complex multi-table relationships and joins
- Fixed schemas with strict validation
- Existing SQL infrastructure and expertise

**What's next:**

With MongoDB fundamentals mastered, explore replica sets (high availability), sharding (horizontal scaling), change streams (real-time data), and integrating MongoDB with Python using PyMongo or with Node.js using the native driver.

**Practice resources:**

- [MongoDB University](https://university.mongodb.com/) - free official courses with certifications
- [MongoDB Documentation](https://docs.mongodb.com/) - comprehensive guides
- [M001: MongoDB Basics](https://university.mongodb.com/courses/M001) - hands-on course

**External resources:**

- [MongoDB Compass](https://www.mongodb.com/products/compass) - GUI for MongoDB
- [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) - free cloud database hosting
- [PyMongo Documentation](https://pymongo.readthedocs.io/) - Python integration

> **Remember:** SQL and NoSQL aren't competitors; they're tools for different jobs. Choose based on your data structure, consistency needs, and scalability requirements. Master both to become a versatile data professional.


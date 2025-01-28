# Performance

<details>
<summary>🟢 Loops</summary>

Most cases i prefer FOR

<details open>
<summary>🟢 Eazy LVL (Current)</summary>

**Key Concepts**:  
- For is faster than While
- Of course you need to use for or while, it depends on the situation
- Also, I write all results in a table

</details>
</details>

| Test Case                          | FOR               | WHILE              | Winner | Difference        |
|------------------------------------|-------------------|--------------------|--------|-------------------|
| For VS While (simple)              | `0.8757s` ✅      | `1.6729s` ❌       | FOR    | `+0.7972s` ⚡     |
| For VS While (math)                | `2.7320s` ✅      | `3.6326s` ❌       | FOR    | `+0.9006s` ⚡     |
| For VS While (list)                | `0.2327s` ✅      | `0.5130s` ❌       | FOR    | `+0.2802s` ⚡     |
| Nested VS While (nested)           | `0.7617s` ✅      | `1.5230s` ❌       | FOR    | `+0.7613s` ⚡     |
| For VS While (break)               | `0.6968s` ✅      | `0.9091s` ❌       | FOR    | `+0.2123s` ⚡     |
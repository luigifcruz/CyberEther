---
title: Typography Test
description: A comprehensive test of all markdown features and styling.
order: 99
category: Testing
---

> [!NOTE]
> This page is for testing documentation typography and features.

## Headings

# Heading 1
## Heading 2
### Heading 3
#### Heading 4

## Paragraphs

This is a regular paragraph with some text. It should have proper line-height and spacing for readability. The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.

This is another paragraph to test spacing between paragraphs. Good typography ensures that text is easy to read and scan.

## Inline Formatting

This text contains **bold text**, *italic text*, and ***bold italic text***.

You can also use `inline code` within paragraphs, like referencing a variable `const foo = "bar"` or a function `calculateTotal()`.

Here is a [link to CyberEther](https://cyberether.net) and another [external link](https://github.com).

## Lists

### Unordered List

- First item
- Second item
- Third item with a longer description that might wrap to multiple lines
- Fourth item
  - Nested item one
  - Nested item two
    - Deeply nested item
- Fifth item

### Ordered List

1. First step
2. Second step
3. Third step
   1. Sub-step one
   2. Sub-step two
4. Fourth step

### Mixed List

1. First ordered item
   - Unordered sub-item
   - Another sub-item
2. Second ordered item

## Code Blocks

Inline code: `npm install cyberether`

```javascript
// JavaScript example
function greet(name) {
    const message = `Hello, ${name}!`;
    console.log(message);
    return message;
}

greet("World");
```

```python
# Python example
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)

for i in range(10):
    print(calculate_fibonacci(i))
```

```bash
# Shell commands
cd ~/projects
git clone https://github.com/example/repo.git
npm install
npm run dev
```

```rust
// Rust example
fn main() {
    let numbers: Vec<i32> = (1..=10).collect();
    let sum: i32 = numbers.iter().sum();
    println!("Sum: {}", sum);
}
```

## Blockquotes

> This is a simple blockquote. It can contain multiple sentences and should be styled distinctly from regular paragraphs.

> This is a multi-line blockquote.
>
> It has multiple paragraphs and maintains proper formatting throughout the entire quote block.

## Callouts

> [!NOTE]
> Useful information that users should know, even when skimming content.

> [!TIP]
> Helpful advice for doing things better or more easily.

> [!IMPORTANT]
> Key information users need to know to achieve their goal.

> [!WARNING]
> Urgent info that needs immediate user attention to avoid problems.

> [!CAUTION]
> Advises about risks or negative outcomes of certain actions.

## Tables

| Feature | Status | Notes |
|---------|--------|-------|
| WebGPU | Stable | Primary rendering backend |
| Vulkan | Stable | Native desktop support |
| Metal | Stable | macOS and iOS |
| WebGL | Deprecated | Legacy fallback |

### Wide Table

| Column 1 | Column 2 | Column 3 | Column 4 | Column 5 |
|----------|----------|----------|----------|----------|
| Data A1 | Data A2 | Data A3 | Data A4 | Data A5 |
| Data B1 | Data B2 | Data B3 | Data B4 | Data B5 |
| Data C1 | Data C2 | Data C3 | Data C4 | Data C5 |

## Horizontal Rule

Content above the rule.

---

Content below the rule.

## Collapsible Sections

<details>
<summary>Click to expand</summary>

This is hidden content that appears when you click the summary.

You can include any markdown here:
- Lists
- **Bold text**
- `code`

</details>

<details>
<summary>Another collapsible section</summary>

More hidden content with a code block:

```javascript
console.log("Hello from inside a details element!");
```

</details>

## Images

Images can be added using standard markdown syntax:

![CyberEther Logo](/cyberether-original.png)

## Combined Elements

Here's a paragraph with **bold**, *italic*, and `code` all together. It also includes a [link](https://example.com).

> [!TIP]
> You can combine callouts with other elements like `inline code` and **bold text**.

1. Step one: Run `npm install`
2. Step two: Configure your settings
3. Step three: Run `npm run dev`

---

End of typography test.

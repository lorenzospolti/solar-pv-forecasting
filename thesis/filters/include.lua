--[[
include.lua: replace a fenced code block of class "include" by the Markdown
files it names, so that generated tables enter the PDF straight from
thesis/tables/ and a rebuilt PDF never shows a stale copy.

    ```{.include}
    tables/master_skill.md
    ```

One path per line, relative to thesis/ (the directory `make manuscript` runs
pandoc from); blank lines are ignored. A missing file becomes a visible
paragraph "[missing include: <path>]" and a warning on stderr. Run before
pandoc-crossref, so the labels of included tables and figures are resolved.
]]

local function warn(msg)
  io.stderr:write("include.lua: " .. msg .. "\n")
end

function CodeBlock(el)
  if not el.classes:includes("include") then
    return nil
  end
  local blocks = pandoc.List()
  for line in el.text:gmatch("[^\r\n]+") do
    local path = line:match("^%s*(.-)%s*$")
    if path ~= "" then
      local fh = io.open(path, "r")
      if fh then
        local text = fh:read("a")
        fh:close()
        blocks:extend(pandoc.read(text, "markdown", PANDOC_READER_OPTIONS).blocks)
      else
        warn("missing include: " .. path)
        blocks:insert(pandoc.Para(pandoc.Inlines("[missing include: " .. path .. "]")))
      end
    end
  end
  return blocks
end

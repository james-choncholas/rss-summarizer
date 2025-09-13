
import feedgen.feed
import os
import feedparser

# Create a feed
fg = feedgen.feed.FeedGenerator()
fg.title('Test Feed')
fg.link(href='http://example.com', rel='alternate')
fg.description('This is a test feed.')

# Add an entry with content
fe1 = fg.add_entry()
fe1.id('1')
fe1.title('Entry 1')
fe1.content('This is the content of entry 1.')

# Add an entry with summary
fe2 = fg.add_entry()
fe2.id('2')
fe2.title('Entry 2')
fe2.summary('This is the summary of entry 2.')

# Generate the feed
fg.rss_file('test_feed.xml', pretty=True)

# Parse the feed
d = feedparser.parse('test_feed.xml')

# Print the entries
for entry in d.entries:
    print(f"Entry title: {entry.title}")
    print(f"Entry content: {entry.get('content')}")
    print(f"Entry summary: {entry.get('summary')}")
    print(f"Entry description: {entry.get('description')}")
    print("-" * 20)

# Clean up
os.remove('test_feed.xml')

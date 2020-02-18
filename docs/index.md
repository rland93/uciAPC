# uci AP Controller Project

![logo]({{ "uciAPC_petr2.png" | absolute_url  }})

Uci AP Controller is an undergraduate research project started in Winter '19-20 at the University of California, Irvine. The faculty advisor for this project is [David Copp](http://engineering.uci.edu/users/david-copp). For more information, view the [project proposal](Project_Proposal.html).

Team members:
+ Michael Sutherland
+ Ashutosh Shah

Documentation of the project is available on the [project wiki](https://github.com/rland93/uciAPC/wiki).

---

Or view the blog posts below:
<ul>
  {% for post in site.posts %}
    <li>
      {{ post.date |  date: '%B %d, %Y' }}: <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>

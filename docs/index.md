# uci AP Controller Project

![logo]({{ "uciAPC_petr2.png" | absolute_url  }})

Uci AP Controller is an undergraduate research project started in Winter '19-20 at the University of California, Irvine. The faculty advisor for this project is [David Copp](http://engineering.uci.edu/users/david-copp). For more information, view the [project proposal](Project_Proposal.html).

We have a [trello board](https://trello.com/b/aRJtnHfg/uciapc) to keep track of new features and changes.

Spring 2020 Team members:
+ Mike Sutherland 
+ Kyle Adamos

Alumni:
+ Ashutosh Shah (W20)

Documentation of the project is available <a href="{{ site.baseurl  }}/documentation/index.html">here</a>. Most documentation is written inline with the code.

---

Or view the blog posts below:
<ul>
  {% for post in site.posts %}
    <li>
      {{ post.date |  date: '%B %d, %Y' }}: <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>

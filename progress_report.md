# Progress Report

|group     | LuckyTiger     |
| :------------- | :------------- |
| topic     | NLP - Speech emotion recognition     |
|attendance   |<ul><li>Vivien Xian</li><li>Zhaocheng Li</li><li>Zhiqing Cen</li><li>Jinlei Ru</li></ul>   |



1. **Why choice this topic?**
    - It can be very hard to identify the actual emotion of people over the phone and on online meeting.
    - Most customer service calls currently are firstly directed to bots instead of human. When a customer is anxious or have an urgent need, talking to a bot which normally cannot solve their request might lead to worse impression about the organisation and make them more anxious. Using speech emotion recognition to identify customer's emotion, and direct the call to human staff when customer emotion reach the set threshold. Speech emotion recognition can also identify angry customer sooner to statify the demand of customer.
    - In a situation which calling the others to explain the actual situation will put the caller in danger, using speech emotion recognition to detect their actual emotion (e.g. fear) can help the phone receiver to have a better understanding of what the caller is experiencing.

2. **What possible/potential limitations of the previous methods for that topic?**
    - Most of the existing speech emotion recognition programs have low accuracy in identifying the actual emotion of the speaker.
        - e.g. in this article [3-D Convolutional Recurrent Neural Networks with Attention Model for Speech Emotion Recognition](images/SER.pdf), speech with happy emotion might mistakenly identify as angry emotion with a 51% probability.
    - Many paper only used audio analysis, but didn't make use of any text analysis.

3. **What potential problems you are going to solve with that topic? You don’t have to be very detail, just general problems, for example, we can solve the performance problem when images quite blur.**
    - We're aiming to solve one of these three problems
    - In some papers, the accuracy rate of speech emotion recognition needs to be improved.
    - Distinguish between emotions that have similar frequency and pitch, such as happy and angry.
    - Use both speech and text recognition in speech emotion recognition to work together to solve this problem.

4. **Which dataset you have selected or you plan to use? Why that one?**
    - We have found these two datasets used in most articles we have read, and the fact that these dataset have a large variaty of speakers, with respective audios and various emotions corresponding to those voices. We will choose one datasets from below.
        - [The Interactive Emotional Dyadic Motion Capture (IEMOCAP) Database](https://sail.usc.edu/iemocap/)
        - [Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://zenodo.org/record/1188976#.YsjoBi8RpQI)

5. **What’s your plan for next two weeks?**
    - More research on papers to improve undertanding of speech emotion recognition logic and steps.
    - Configure the environment needed for the program and run code provided in articles.
    - Start implementation of the problem we are going to solve.
    - Start on writing the draft of final report.

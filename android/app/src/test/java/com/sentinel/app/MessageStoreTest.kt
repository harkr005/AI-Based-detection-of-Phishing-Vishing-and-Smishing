package com.sentinel.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

// One page ordered by time replaced three ordered by verdict, so this ordering
// is now the only thing deciding what the user sees first.
class MessageStoreTest {

    @Before
    fun reset() = MessageStore.clear()

    @Test
    fun `newest first, whatever order they were written in`() {
        // The inbox backfill reads DATE DESC while the live receiver appends as
        // SMS land, so the store is filled in both directions and insertion
        // order means nothing.
        MessageStore.add("HDFCBK", "newest", 3_000L)
        MessageStore.add("VM-ICICI", "oldest", 1_000L)
        MessageStore.add("Amazon", "middle", 2_000L)

        assertEquals(
            listOf("newest", "middle", "oldest"),
            MessageStore.all().map { it.body },
        )
    }

    @Test
    fun `two in the same millisecond keep a stable order`() {
        val first = MessageStore.add("A", "first", 5_000L)!!
        val second = MessageStore.add("B", "second", 5_000L)!!

        // id descending breaks the tie, so a repaint cannot swap them.
        assertEquals(listOf(second.id, first.id), MessageStore.all().map { it.id })
        assertEquals(MessageStore.all().map { it.id }, MessageStore.all().map { it.id })
    }

    @Test
    fun `a failed scan is on the page, and is not mistaken for a pending one`() {
        val failed = MessageStore.add("Unknown", "could not be scanned", 1_000L)!!
        failed.category = CATEGORY_ERROR
        val pending = MessageStore.add("Other", "still going", 2_000L)!!

        // It used to have no section to appear in at all.
        assertEquals(listOf("still going", "could not be scanned"), MessageStore.all().map { it.body })
        assertTrue(failed.failed)
        assertFalse(pending.failed)
    }

    // The live receiver and the inbox sync both capture the same SMS, and
    // they timestamp it from different clocks — so this, not the hash column,
    // is what stops one message appearing on the page twice.
    @Test
    fun `the same SMS from both capture paths is one message`() {
        val live = MessageStore.add("AX-ICICIT-S", "Your OTP is 4821", 1_700_000_000_000L)
        // The provider row for the same SMS: the phone wrote it four seconds
        // after the SMSC stamped it.
        val synced = MessageStore.add("AX-ICICIT-S", "Your OTP is 4821", 1_700_000_004_000L)

        assertNotNull(live)
        assertNull("the sync must not re-add what the receiver already caught", synced)
        assertEquals(1, MessageStore.all().size)

        // A day later is a different message, not a late copy of this one.
        val nextDay = MessageStore.add("AX-ICICIT-S", "Your OTP is 4821", 1_700_086_400_001L)
        assertNotNull(nextDay)
        assertEquals(2, MessageStore.all().size)
    }

    @Test
    fun `unpairing empties the store`() {
        MessageStore.add("HDFCBK", "belongs to the old account", 1_000L)
        MessageStore.clear()
        assertEquals(emptyList<ScannedMessage>(), MessageStore.all())
    }
}
